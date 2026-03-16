# MIL Training — Quick Reference

---

## Memory Test

Gerçekçi training memory ölçümü (gradyan + optimizer state dahil).

```bash
# Tüm aggregatorler, 224 ve 384, birkaç frame sayısı
python src/test_memory.py --size 224 384 --frames 300 400 500 600

# Sadece belirli aggregatorler
python src/test_memory.py --size 224 384 --frames 500 \
    --agg mean mh_gated4 transmil1d8_3_w64

# Farklı GPU üzerinde test
python src/test_memory.py --size 224 384 --frames 500 --gpu 1
```

---

## Cache Generation for Video Files

python src/precompute_cache.py --img_size 384 

---

## Ablation — Size Seçimi (Aşama 1)

WACV'deki en iyi 3 aggregator ile 224 vs 384 karşılaştırması.
`configs/ablation.yaml` içinde aktif aggregatorleri ayarlayın.

```bash
# Her iki size, default 600 frame
python src/ablation_run_all.py --sizes 224 384 --n_frames 600

# 384 için daha az frame (memory'e göre ayarlayın)
python src/ablation_run_all.py --sizes 224 384 --n_frames 500

# Sadece 384
python src/ablation_run_all.py --sizes 384 --n_frames 500

# Özel YAML ile
python src/ablation_run_all.py --sizes 224 384 \
    --poolers_yaml configs/ablation.yaml
```

---

## Full Training — Aggregator Ablasyonu (Aşama 2)

En iyi size bulunduktan sonra `configs/poolers.yaml`'daki tüm
aggregatorleri o size ile eğitin.

```bash
# 224 ile full run (tüm aktif aggregatorler poolers.yaml'dan)
python src/run_all.py --img_size 224 --n_frames 600

# 384 ile full run
python src/run_all.py --img_size 384 --n_frames 500

# Özel YAML ile
python src/run_all.py --img_size 224 --n_frames 600 \
    --poolers_yaml configs/poolers.yaml
```

---

## Run Klasörü İsimlendirmesi

```
runs/
  mh_gated4_s224_f600/
    fold00/  fold01/  fold02/  fold03/  fold04/
  mh_gated4_s384_f500/
    fold00/  ...
  tmil1d_w64_s224_f600/
    fold00/  ...
```

Her fold klasöründe:
- `train_log.csv` — epoch başına loss/mae/lr
- `train_log.txt` — human-readable log
- `best.pt` — en iyi checkpoint
- `val_predicted_true_values.csv`
- `test_predicted_true_values.csv`
- `config.json` — o run'ın tam konfigürasyonu

---

## configs/ablation.yaml Örneği

```yaml
experiments:
  - name: mh_gated4
    aggregator: mh_gated4

  - name: tconv32_gated8
    aggregator: temporal_conv32h8

  - name: tmil1d_w64
    aggregator: transmil1d8_3_w64
```

---

## configs/poolers.yaml — Aggregator Aktif Etme

Satır başındaki `#` kaldırılarak aggregator aktif edilir:

```yaml
experiments:
  - name: mean
    aggregator: mean

# - name: gru        ← bu pasif
#   aggregator: gru
```

---

## Önerilen Akış

```
1. python src/test_memory.py --size 224 384 --frames 500 600
      → Hangi (size, n_frames) güvenli? → belirle

2. configs/ablation.yaml → en iyi 3 aggregatoru yaz

3. python src/ablation_run_all.py --sizes 224 384 --n_frames <N>
      → Hangi size daha iyi? → belirle

4. configs/poolers.yaml → tüm aggregatorleri aktif et

5. python src/run_all.py --img_size <best_size> --n_frames <N>
      → Hangi aggregator en iyi? → belirle
```


# python src/run_all.py --img_size 224 --n_frames 200
# python src/run_all.py --img_size 384 --n_frames 200