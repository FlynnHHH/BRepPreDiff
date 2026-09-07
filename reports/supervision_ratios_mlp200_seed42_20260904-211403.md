# Multi-supervision-ratio fine-tuning results

Seed 42; MLP full fine-tuning for 200 epochs; full validation every 5 epochs and
unchanged test sets; best checkpoint selected by
validation accuracy. Ratios count labeled CAD models.
The 100% endpoints reuse matching earlier runs with validation every epoch; all ratios
below 100% are newly run with validation every 5 epochs.
Encoder checkpoint: `runs/inductive_lr1e4_encoder/pretrain/20260901-115311_inductive9_no_fab_encoder_lr1e4_e100_seed42_20260901/checkpoints/last.pt`.

| Benchmark | Ratio | Train CADs | Test CADs | Best epoch | Accuracy (%) | Macro-F1 (%) | Weighted-F1 (%) | mIoU (%) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| CADSynth | 0.1% | 80 | 9993 | 195 | 77.6475 | 53.4950 | 77.4930 | 38.9174 |
| CADSynth | 0.5% | 400 | 9993 | 190 | 93.3698 | 84.8142 | 93.3276 | 75.7522 |
| CADSynth | 1% | 800 | 9993 | 200 | 96.0554 | 90.8283 | 95.9952 | 84.6260 |
| CADSynth | 1.5% | 1200 | 9993 | 195 | 96.9104 | 93.0528 | 96.8977 | 87.8077 |
| CADSynth | 2% | 1599 | 9993 | 185 | 97.5773 | 94.6103 | 97.5626 | 90.3057 |
| CADSynth | 3% | 2399 | 9993 | 195 | 98.0475 | 95.8278 | 98.0346 | 92.3281 |
| CADSynth | 100% | 79941 | 9993 | 62 | 99.5730 | 99.2922 | 99.5726 | 98.5993 |
| MFCAD++ | 0.1% | 42 | 8949 | 180 | 61.8662 | 44.1151 | 61.8953 | 31.2288 |
| MFCAD++ | 0.5% | 209 | 8949 | 175 | 88.6911 | 80.5130 | 88.6478 | 69.4039 |
| MFCAD++ | 1% | 418 | 8949 | 180 | 92.5936 | 87.2185 | 92.5261 | 78.7103 |
| MFCAD++ | 1.5% | 627 | 8949 | 185 | 94.0531 | 89.7275 | 93.9908 | 82.4273 |
| MFCAD++ | 2% | 836 | 8949 | 200 | 95.0201 | 91.3115 | 95.0028 | 84.8344 |
| MFCAD++ | 3% | 1253 | 8949 | 200 | 95.8064 | 92.7277 | 95.7821 | 87.0463 |
| MFCAD++ | 100% | 41766 | 8949 | 185 | 99.4122 | 99.0716 | 99.4120 | 98.1722 |
| MFInstSeg | 0.1% | 50 | 6250 | 195 | 70.7889 | 48.5275 | 69.9063 | 36.5545 |
| MFInstSeg | 0.5% | 250 | 6250 | 180 | 90.0086 | 81.7060 | 89.9595 | 71.4395 |
| MFInstSeg | 1% | 500 | 6250 | 190 | 92.8958 | 86.6848 | 92.8161 | 78.1592 |
| MFInstSeg | 1.5% | 750 | 6250 | 200 | 94.5236 | 90.1211 | 94.4636 | 83.1675 |
| MFInstSeg | 2% | 1000 | 6250 | 190 | 94.9324 | 91.0120 | 94.8977 | 84.5837 |
| MFInstSeg | 3% | 1500 | 6250 | 190 | 95.8176 | 92.4824 | 95.7966 | 86.8140 |
| MFInstSeg | 100% | 49996 | 6250 | 194 | 99.2767 | 98.7694 | 99.2758 | 97.5963 |

## Data-efficiency summary

Gaps are 3% minus 100% in percentage points; negative values indicate remaining
full-supervision headroom.

| Benchmark | Accuracy gain 0.1%→0.5% | Accuracy at 3% | Accuracy gap | Macro-F1 at 3% | Macro-F1 gap |
|---|---:|---:|---:|---:|---:|
| CADSynth | +15.7224 | 98.0475 | -1.5255 | 95.8278 | -3.4644 |
| MFCAD++ | +26.8248 | 95.8064 | -3.6058 | 92.7277 | -6.3438 |
| MFInstSeg | +19.2197 | 95.8176 | -3.4591 | 92.4824 | -6.2870 |
