
# run_all.py
"""
Global job queue scheduler for multi-GPU 5-fold cross-validation.
MIL-Lab edition: runs ABMIL, TransMIL, CLAMSB, DSMIL.

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
N_FOLDS   = 5                       # must match CFG.n_folds
N_EPOCHS  = 100                     # must match CFG.epochs — used for completion check
SLEEP_SEC = 10                      # polling interval (seconds)
FOLD_INDICES = None                 # None = all folds; list e.g. [0] = fold0 only

RUNS_DIR = Path(
    "/research/projects/Sahika/projects/liver_PDFF"
    "/clean_code_final_random_train/millab_training/runs"
)

IMG_SIZE  = 384
N_FRAMES  = 75

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def ts() -> str:
    return datetime.now().strftime("%H:%M:%S")


def load_experiments() -> list[dict]:
    with open(POOLERS_YAML, "r") as f:
        cfg = yaml.safe_load(f)
    exps = cfg.get("experiments", [])
    if not exps:
        raise RuntimeError(f"No experiments found in {POOLERS_YAML}")
    return exps


def fold_is_complete(run_name: str, fold_idx: int) -> bool:
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
    queue: deque[dict] = deque()
    skipped = 0

    folds_to_run = FOLD_INDICES if FOLD_INDICES is not None else list(range(N_FOLDS))

    for exp in exps:
        agg      = exp["aggregator"]
        run_name = exp.get("name", agg)
        run_name_sized = f"{run_name}_s{IMG_SIZE}_f{N_FRAMES}"

        for fidx in folds_to_run:
            if fold_is_complete(run_name_sized, fidx):
                print(f"[SKIP]   {run_name_sized}/fold{fidx:02d} — already complete.")
                skipped += 1
            else:
                queue.append({"aggregator": agg, "run_name": run_name, "fold_idx": fidx})

    print(f"\n[QUEUE] {len(queue)} jobs to run, {skipped} already complete.\n")
    return queue


def launch_job(gpu_id: str, job: dict) -> subprocess.Popen:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = gpu_id

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
    gpu_worker: dict = {g: None for g in GPUS}

    def fill_idle_gpus() -> None:
        for gpu_id in GPUS:
            if gpu_worker[gpu_id] is None and queue:
                job  = queue.popleft()
                proc = launch_job(gpu_id, job)
                gpu_worker[gpu_id] = (proc, job)

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
                continue

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

        fill_idle_gpus()

    print(f"\n{'='*70}")
    print(f"ALL JOBS FINISHED — {total_done} succeeded, {total_fail} failed.")
    print(f"{'='*70}\n")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args():
    import argparse
    p = argparse.ArgumentParser(description="Multi-GPU MIL-Lab job scheduler")
    p.add_argument("--img_size", type=int, default=384,
                   help="Frame size (must match a precomputed cache). Default: 384")
    p.add_argument("--n_frames", type=int, default=75,
                   help="Number of frames to sample per video. Default: 75")
    p.add_argument("--folds", type=str, default="0",
                   help="Comma-separated fold indices to run, or 'all'. Default: 0")
    p.add_argument("--poolers_yaml", type=str, default=None,
                   help="Override path to poolers YAML config")
    return p.parse_args()


def main() -> None:
    global IMG_SIZE, N_FRAMES, POOLERS_YAML, FOLD_INDICES

    args = parse_args()
    IMG_SIZE = args.img_size
    N_FRAMES = args.n_frames
    if args.poolers_yaml:
        POOLERS_YAML = Path(args.poolers_yaml)
    if args.folds.lower() == "all":
        FOLD_INDICES = None
    else:
        FOLD_INDICES = [int(x.strip()) for x in args.folds.split(",")]

    folds_str = "all" if FOLD_INDICES is None else str(FOLD_INDICES)
    print(f"[CONFIG] img_size={IMG_SIZE}  n_frames={N_FRAMES}  folds={folds_str}  poolers_yaml={POOLERS_YAML}\n")

    exps  = load_experiments()
    queue = build_job_queue(exps)

    if not queue:
        print("Nothing to do — all folds already complete.")
        return

    print(f"Starting scheduler with {len(GPUS)} GPUs.\n")
    run_scheduler(queue)


if __name__ == "__main__":
    main()


# Trend check: fold0 only, 384px, 75 frames (one job per GPU — all 4 run in parallel)
# python /research/projects/Sahika/projects/liver_PDFF/clean_code_final_random_train/millab_training/src/run_all.py --img_size 384 --n_frames 75 --folds 0
#
# Full 5-fold run:
# python /research/projects/Sahika/projects/liver_PDFF/clean_code_final_random_train/millab_training/src/run_all.py --img_size 384 --n_frames 75 --folds all
