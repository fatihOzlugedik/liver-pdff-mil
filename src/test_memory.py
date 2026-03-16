"""
test_memory.py — realistic GPU memory probe with gradients + optimizer state

Usage:
  python test_memory.py
  python test_memory.py --size 224 384 --frames 300 500 600
  python test_memory.py --agg mean mh_gated4 transmil1d8_3_w64 --size 224 384 --frames 500
"""

import argparse, sys, gc
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from sympy import python
import torch

BACKBONE = "swin_tiny_patch4_window7_224.ms_in1k"

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--size",   type=int, nargs="+", default=[224, 384])
    p.add_argument("--frames", type=int, nargs="+", default=[300, 500, 600])
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
    p.add_argument("--gpu",    type=int, default=0)
    p.add_argument("--grad_accum", type=int, default=8,  help="match CFG.grad_accum")
    return p.parse_args()


def probe(agg, img_size, n_frames, grad_accum, device):
    from config import CFG
    from model import MILModel

    cfg = CFG(aggregator=agg, run_name=f"memtest_{agg}")
    cfg.backbone  = BACKBONE
    cfg.img_size  = img_size
    cfg.grad_accum = grad_accum

    model = MILModel(cfg).to(device)
    model.train()

    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    loss_fn = torch.nn.L1Loss()

    bag = torch.randn(n_frames, 1, img_size, img_size, device=device)
    y   = torch.tensor([5.0], device=device)

    torch.cuda.reset_peak_memory_stats(device)
    try:
        opt.zero_grad()
        # simulate grad_accum steps — worst case is first step
        y_hat = model(bag)
        loss  = loss_fn(y_hat, y) / grad_accum
        loss.backward()
        opt.step()

        peak_mb = torch.cuda.max_memory_allocated(device) / 1024**2
        status = f"{peak_mb:7.0f} MB"
    except torch.cuda.OutOfMemoryError:
        status = "   OOM ❌"
        peak_mb = None
    except RuntimeError as e:
        status = f"   ERR: {str(e)[:40]}"
        peak_mb = None

    del model, opt, bag, y
    torch.cuda.empty_cache()
    gc.collect()
    return status, peak_mb


def main():
    args   = parse_args()
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    total_vram = torch.cuda.get_device_properties(device).total_memory / 1024**2
    print(f"\nDevice: {device}  |  Total VRAM: {total_vram:.0f} MB")
    print(f"Backbone: {BACKBONE}  |  grad_accum={args.grad_accum}  (realistic training mode)\n")

    header = f"{'Aggregator':<28} {'size':>5} {'frames':>7}  {'Peak VRAM':>12}"
    print(header)
    print("-" * len(header))

    for agg in args.agg:
        for size in args.size:
            for nf in args.frames:
                status, _ = probe(agg, size, nf, args.grad_accum, device)
                print(f"{agg:<28} {size:>5} {nf:>7}  {status:>12}")
        print()


if __name__ == "__main__":
    main()

# python src/test_memory.py --size 512 384 224 --frames 200 300 400 500 600

# python /research/projects/Sahika/projects/liver_PDFF/clean_code_final_random_train/src/test_memory.py --size 512  --frames 110 120
# python /research/projects/Sahika/projects/liver_PDFF/clean_code_final_random_train/src/test_memory.py --size 384  --frames 200 210 225 250