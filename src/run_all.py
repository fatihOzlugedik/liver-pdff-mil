
# run_all.py
"""
Global job queue scheduler for multi-GPU 5-fold cross-validation.

Strategy:
  - Build a single flat queue of ALL (model, fold) jobs upfront.
  - Maintain a pool of 4 GPU workers.
  - As soon as any GPU finishes, assign it the next job from the queue.
  - No GPU ever idles while jobs remain; no GPU ever runs two jobs at once.

Resume logic:
  - A fold is considered complete if its train_log.csv has >= N_EPOCHS rows.
  - Incomplete/crashed folds have their output directory wiped and are re-queued.
"""

import os
import sys
import time
import subprocess
import yaml
import shutil
import pandas as pd
from pathlib import Path
from collections import deque
from datetime import datetime

# ---------------------------------------------------------------------------
# Paths & constants
# ---------------------------------------------------------------------------
THIS_DIR     = Path(__file__).resolve().parent
MAIN_PY      = THIS_DIR / "main.py"
POOLERS_YAML = THIS_DIR.parent / "configs" / "poolers.yaml"

GPUS      = ["0", "1", "2", "3"]   # GPU IDs available on the machine
N_FOLDS   = 1                       # must match CFG.n_folds
N_EPOCHS  = 100                     # must match CFG.epochs — used for completion check
SLEEP_SEC = 10                      # polling interval (seconds)

RUNS_DIR = Path(
    "/research/projects/Sahika/projects/liver_PDFF"
    "/clean_code_final_random_train/runs_without_test"
)

# IMG_SIZE is now set via --img_size CLI argument (default: 384)
IMG_SIZE  = 384                     # overridden by parse_args()

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def ts() -> str:
    """Return a compact HH:MM:SS timestamp."""
    return datetime.now().strftime("%H:%M:%S")


def load_experiments() -> list[dict]:
    with open(POOLERS_YAML, "r") as f:
        cfg = yaml.safe_load(f)
    exps = cfg.get("experiments", [])
    if not exps:
        raise RuntimeError(f"No experiments found in {POOLERS_YAML}")
    return exps


def fold_is_complete(run_name: str, fold_idx: int) -> bool:
    """
    Return True only if the fold already has a train_log.csv with >= N_EPOCHS rows.
    Otherwise wipe the fold directory and return False so it will be re-run.
    """
    fold_dir = RUNS_DIR / run_name / f"fold{fold_idx:02d}"
    log_csv  = fold_dir / "train_log.csv"

    if not log_csv.exists():
        return False

    try:
        df = pd.read_csv(log_csv)
        if len(df) >= N_EPOCHS:
            return True
        print(
            f"[RESTART] {run_name}/fold{fold_idx:02d}: "
            f"only {len(df)}/{N_EPOCHS} epochs completed — wiping and re-queuing."
        )
    except Exception as exc:
        print(
            f"[RESTART] {run_name}/fold{fold_idx:02d}: "
            f"could not read log ({exc}) — wiping and re-queuing."
        )

    shutil.rmtree(fold_dir, ignore_errors=True)
    return False


def build_job_queue(exps: list[dict]) -> deque[dict]:
    """
    Iterate over all experiments x folds and collect jobs that are not yet complete.
    Returns a deque of dicts: {aggregator, run_name, fold_idx}.
    """
    queue: deque[dict] = deque()
    skipped = 0

    for exp in exps:
        agg      = exp["aggregator"]
        run_name = exp.get("name", agg)
        run_name_sized = f"{run_name}_s{IMG_SIZE}_f{N_FRAMES}"

        for fidx in range(N_FOLDS):
            if fold_is_complete(run_name_sized, fidx):
                print(f"[SKIP]   {run_name_sized}/fold{fidx:02d} — already complete.")
                skipped += 1
            else:
                queue.append({"aggregator": agg, "run_name": run_name, "fold_idx": fidx})

    print(f"\n[QUEUE] {len(queue)} jobs to run, {skipped} already complete.\n")
    return queue


def launch_job(gpu_id: str, job: dict) -> subprocess.Popen:
    """Spawn main.py for a single (model, fold) job on the given GPU."""
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = gpu_id

    # Append img_size to run_name so 224 and 384 runs are saved in separate dirs
    run_name_sized = f"{job['run_name']}_s{IMG_SIZE}"
    cmd = [
        sys.executable, str(MAIN_PY),
        "--aggregator", job["aggregator"],
        "--n_folds",    str(N_FOLDS),
        "--fold_idx",   str(job["fold_idx"]),
        "--run_name",   run_name_sized,
        "--img_size",   str(IMG_SIZE),
        "--n_frames",   str(N_FRAMES),
    ]

    print(
        f"[LAUNCH {ts()}] GPU={gpu_id}  "
        f"agg={job['aggregator']}  "
        f"fold={job['fold_idx'] + 1}/{N_FOLDS}  "
        f"run={job['run_name']}"
    )
    return subprocess.Popen(cmd, env=env, cwd=str(THIS_DIR))


# ---------------------------------------------------------------------------
# Scheduler
# ---------------------------------------------------------------------------

def run_scheduler(queue: deque[dict]) -> None:
    """
    GPU worker-pool loop.

    gpu_worker[gpu_id] = (Popen, job_dict)  while a job is running
                       = None               while the GPU is free
    """
    gpu_worker: dict = {g: None for g in GPUS}

    def fill_idle_gpus() -> None:
        for gpu_id in GPUS:
            if gpu_worker[gpu_id] is None and queue:
                job  = queue.popleft()
                proc = launch_job(gpu_id, job)
                gpu_worker[gpu_id] = (proc, job)

    # Initial fill — start one job per GPU
    fill_idle_gpus()

    total_done = 0
    total_fail = 0

    while any(w is not None for w in gpu_worker.values()):
        time.sleep(SLEEP_SEC)

        for gpu_id in GPUS:
            slot = gpu_worker[gpu_id]
            if slot is None:
                continue

            proc, job = slot
            rc = proc.poll()
            if rc is None:
                continue  # still running

            # Job finished
            if rc == 0:
                total_done += 1
                print(
                    f"[DONE  {ts()}] GPU={gpu_id}  "
                    f"agg={job['aggregator']}  fold={job['fold_idx'] + 1}/{N_FOLDS}  "
                    f"run={job['run_name']}  rc=0 OK"
                )
            else:
                total_fail += 1
                print(
                    f"[FAIL  {ts()}] GPU={gpu_id}  "
                    f"agg={job['aggregator']}  fold={job['fold_idx'] + 1}/{N_FOLDS}  "
                    f"run={job['run_name']}  rc={rc}"
                )

            gpu_worker[gpu_id] = None

        # Assign waiting jobs to newly freed GPUs
        fill_idle_gpus()

    print(f"\n{'='*70}")
    print(f"ALL JOBS FINISHED — {total_done} succeeded, {total_fail} failed.")
    print(f"{'='*70}\n")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args():
    import argparse
    p = argparse.ArgumentParser(description="Multi-GPU MIL job scheduler")
    p.add_argument("--img_size", type=int, default=384,
                   help="Frame size to use (must match a precomputed cache). Default: 384")
    p.add_argument("--n_frames", type=int, default=175,
                   help="Number of frames to sample per video. Default: 175")
    p.add_argument("--poolers_yaml", type=str, default=None,
                   help="Override path to poolers YAML config")
    return p.parse_args()


def main() -> None:
    global IMG_SIZE, N_FRAMES, POOLERS_YAML

    args = parse_args()
    IMG_SIZE = args.img_size
    N_FRAMES = args.n_frames
    if args.poolers_yaml:
        POOLERS_YAML = Path(args.poolers_yaml)

    print(f"[CONFIG] img_size={IMG_SIZE}  n_frames={N_FRAMES}  poolers_yaml={POOLERS_YAML}\n")

    exps  = load_experiments()
    queue = build_job_queue(exps)

    if not queue:
        print("Nothing to do — all folds already complete.")
        return

    print(f"Starting scheduler with {len(GPUS)} GPUs.\n")
    run_scheduler(queue)


if __name__ == "__main__":
    main()


# python /research/projects/Sahika/projects/liver_PDFF/clean_code_final_random_train/src/run_all.py --img_size 384 --n_frames 225
# python /research/projects/Sahika/projects/liver_PDFF/clean_code_final_random_train/src/run_all.py --img_size 512 --n_frames 100

# CUDA_VISIBLE_DEVICES=0 python /research/projects/Sahika/projects/liver_PDFF/clean_code_final_random_train/src/main.py --img_size 384 --n_frames 175 --fold_idx 1 --aggregator mean
# CUDA_VISIBLE_DEVICES=1 python /research/projects/Sahika/projects/liver_PDFF/clean_code_final_random_train/src/main.py --img_size 384 --n_frames 175 --fold_idx 1 --aggregator max
# CUDA_VISIBLE_DEVICES=2 python /research/projects/Sahika/projects/liver_PDFF/clean_code_final_random_train/src/main.py --img_size 384 --n_frames 175 --fold_idx 1 --aggregator mh_gated4
# CUDA_VISIBLE_DEVICES=3 python /research/projects/Sahika/projects/liver_PDFF/clean_code_final_random_train/src/main.py --img_size 384 --n_frames 175 --fold_idx 1 --aggregator temporal_conv32h4
# CUDA_VISIBLE_DEVICES=0 python /research/projects/Sahika/projects/liver_PDFF/clean_code_final_random_train/src/main.py --img_size 384 --n_frames 175 --fold_idx 1 --aggregator abmil

    p.add_argument("--agg",    type=str, nargs="+",
                   default=[
                       "mean", "max",
                       "attention",
                       "transf4", "transf8",
                       "gated", "mh_gated4", "mh_gated8", "mh_gated16",
                       "temporal_conv16h4", "temporal_conv32h4", "temporal_conv32h8", "temporal_conv64h8",
                       "gru",
                       "transmil1d4_2_w64", "transmil1d8_2_w64", "transmil1d8_3_w32", "transmil1d8_3_w64",
                       "transmil1d8_3", "transmil1d8_3_notpeg", "transmil1d8_6_w64", "transmil1d12_4_w96",
                   ])