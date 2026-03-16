#!/usr/bin/env python3
# inspect_attention.py
"""
Verilen best.pt dosyasını yükler, CFG'deki test setinde ilerler ve
(eğer model AttentionMIL veya GatedAttentionMIL kullanıyorsa) her video
için en yüksek 10 attention skorunu raporlar.

Kullanım:
    python inspect_attention.py /path/to/best.pt
Çıktı:
    • konsola kısa özet
    • aynı klasöre  test_attention_top10.csv
"""

import sys, json, heapq, torch, pandas as pd
from pathlib import Path
from typing import List, Dict

# -------------------------------- yerel importlar ----------------------
ROOT = Path(__file__).resolve().parent
sys.path.append(str(ROOT))          # model.py, data.py, config.py vb.

from config import CFG, set_seed
from data   import build_loaders
from model  import MILModel
# ----------------------------------------------------------------------

def patch_attention(mod):
    """AttentionMIL veya GatedAttentionMIL katmanını α kaydeder hâle getirir."""
    if mod.__class__.__name__ == "AttentionMIL":
        def fwd(self, x):
            self.last_alpha = torch.softmax(self.att(x), 0)
            return (self.last_alpha * x).sum(0)
    else:  # Gated
        def fwd(self, x):
            a = torch.sigmoid(self.U(x)) * torch.tanh(self.V(x))
            self.last_alpha = torch.softmax(self.w(a), 0)
            return (self.last_alpha * x).sum(0)
    mod.forward = fwd.__get__(mod, mod.__class__)

def collect_attention(model: MILModel, test_dl, device="cuda") -> pd.DataFrame:
    rows: List[Dict] = []

    # ilgili tüm attention modüllerini patch'le
    attn_mods = [m for m in model.modules()
                 if m.__class__.__name__ in {"AttentionMIL","GatedAttentionMIL"}]
    for m in attn_mods:
        patch_attention(m)

    model.eval()
    with torch.no_grad():
        for bag, label, pid in test_dl:          # bs = 1
            pid = int(pid)
            bag = bag.to(device)
            _   = model(bag)                     # forward

            # çok-head'li ise ortalama al
            alphas = [m.last_alpha.squeeze(1).cpu() for m in attn_mods]
            alpha  = torch.stack(alphas).mean(0)  # (T,)

            top10  = heapq.nlargest(
                        10, [(float(a), idx) for idx, a in enumerate(alpha)]
                     )

            rows.append({
                "patient_ID": pid,
                "gt_pdff"   : float(label),
                "top_frames": [(idx, round(score,4)) for score, idx in top10]
            })
    return pd.DataFrame(rows)

# ----------------------------------------------------------------------
def main():
    if len(sys.argv) != 2:
        sys.exit("Kullanım:  python inspect_attention.py /path/to/best.pt")

    ckpt_path = Path(sys.argv[1]).expanduser()
    if not ckpt_path.exists():
        sys.exit(f"❌  Bulunamadı: {ckpt_path}")

    # -------- CFG'i varsayılanlarla oluştur (config.py'deki değerler) ---
    cfg = CFG()     
    set_seed(cfg.seed)                     # tüm parametreler hazır
    cfg.device = "cuda" if torch.cuda.is_available() else "cpu"

    # -------- model + ağırlıklar ---------------------------------------
    model = MILModel(cfg).to(cfg.device)
    model.load_state_dict(torch.load(ckpt_path, map_location=cfg.device))

    # -------- test loader ----------------------------------------------
    _, _, test_dl = build_loaders(cfg)

    # -------- attention toplama ----------------------------------------
    df = collect_attention(model, test_dl, cfg.device)

    out_csv = ckpt_path.with_name("test_attention_top10.csv")
    df.to_csv(out_csv, index=False)
    print("\nİlk 5 satır:")
    print(df.head().to_string(index=False))
    print("\nKaydedildi →", out_csv)

if __name__ == "__main__":
    main()
