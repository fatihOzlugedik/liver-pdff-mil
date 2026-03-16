"""
precompute_cache.py

Reads all videos once and saves each as a .npy file of shape (T, img_size, img_size)
uint8 grayscale — exactly the format consumed by LiverPDFFDataset.

Cache is stored per img_size so multiple sizes can coexist:
  frame_cache_224/  →  (T, 224, 224) uint8
  frame_cache_384/  →  (T, 384, 384) uint8

Usage:
    python precompute_cache.py                  # default 224
    python precompute_cache.py --img_size 384   # for 384x384
"""

import sys
import argparse
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd
from PIL import Image
from config import CFG

try:
    from decord import VideoReader, cpu as decord_cpu
    DECORD_AVAILABLE = True
except ImportError:
    DECORD_AVAILABLE = False
    from torchvision.io import read_video
    print("[WARN] decord not found, using torchvision (slower)")


def _to_sec(val) -> float:
    val     = float(val)
    minutes = int(val)
    seconds = round((val - minutes) * 100)
    return float(minutes * 60 + seconds)


def load_and_preprocess_segment(vid_path: str,
                                start_sec: float,
                                end_sec: float,
                                img_size: int) -> np.ndarray:
    """
    Load video segment, resize to (img_size, img_size) grayscale.
    Returns (T, img_size, img_size) uint8.
    """
    if DECORD_AVAILABLE:
        vr      = VideoReader(vid_path, ctx=decord_cpu(0))
        fps     = vr.get_avg_fps()
        total   = len(vr)
        start_f = max(0, int(start_sec * fps))
        end_f   = min(total - 1, int(end_sec * fps))
        if end_f <= start_f:
            end_f = min(start_f + 1, total - 1)
        raw = vr.get_batch(list(range(start_f, end_f + 1))).asnumpy()
    else:
        frames, _, _ = read_video(vid_path, start_pts=start_sec,
                                  end_pts=end_sec, pts_unit="sec")
        raw = frames.numpy()

    target = (img_size, img_size)
    processed = np.stack([
        np.array(
            Image.fromarray(frame).convert("L").resize(target, Image.BILINEAR)
        )
        for frame in raw
    ], axis=0)  # (T, img_size, img_size) uint8

    return processed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--img_size", type=int, default=224,
                        help="Frame size to cache (default: 224)")
    args = parser.parse_args()

    cfg           = CFG(img_size=args.img_size)
    cache_dir     = cfg.cache_dir   # frame_cache_{img_size}/
    cache_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(cfg.csv_path)
    print(f"img_size={args.img_size}  |  {len(df)} videos  →  {cache_dir}\n")

    ok, skip       = 0, 0
    total_saved_mb = 0.0

    for i, row in df.iterrows():
        vid      = str(row[cfg.video_id_col])
        vid_path = str(row[cfg.video_path_col])
        start_t  = _to_sec(row[cfg.start_col])
        end_t    = _to_sec(row[cfg.end_col])
        if end_t < start_t:
            start_t, end_t = end_t, start_t

        out_path = cache_dir / f"{vid}.npy"

        if out_path.exists():
            size_mb = out_path.stat().st_size / 1024**2
            print(f"  [{i+1}/{len(df)}] {vid}: already cached ({size_mb:.0f} MB), skipping.")
            ok += 1
            continue

        try:
            frames  = load_and_preprocess_segment(vid_path, start_t, end_t, args.img_size)
            if frames.shape[0] == 0:
                raise RuntimeError("0 frames returned")

            np.save(str(out_path), frames)
            size_mb = out_path.stat().st_size / 1024**2
            total_saved_mb += size_mb
            print(f"  [{i+1}/{len(df)}] {vid}: {frames.shape} → {size_mb:.0f} MB saved.")
            ok += 1

        except Exception as exc:
            print(f"  [{i+1}/{len(df)}] [SKIP] {vid}: {exc}")
            skip += 1

    print(f"\nDone: {ok} OK, {skip} skipped.")
    print(f"Total written: {total_saved_mb/1024:.2f} GB")
    print(f"Cache location: {cache_dir}")


if __name__ == "__main__":
    main()