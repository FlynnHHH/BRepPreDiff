# 50-epoch encoder few-shot results

Seed 42; MLP and DiffLoss heads; 100 fine-tuning epochs; full validation split selects the best
checkpoint, which is evaluated once on the unchanged test split.
Classification uses Mean+Max graph pooling. `DiffLoss` uses the cross-task aligned setting:
x_start prediction, epsilon loss weight 0, one DDIM sampling step, and temperature 0.75.
`DiffLoss-xstart-epsilon` is the retained legacy SolidLetters variant (epsilon weight 0.5,
25 DDIM sampling steps, and temperature 1.0) and is excluded from the aligned head comparison.

| Task | Head | Shot | Unique train CADs | Test CADs | Best epoch | Accuracy (%) | Macro F1 (%) | Macro IoU (%) |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| blendit_seg | DiffLoss | 10 | 30 | 1787 | 60 | 94.4989 | 84.5118 | 74.3500 |
| blendit_seg | MLP | 10 | 30 | 1787 | 100 | 95.2913 | 87.4511 | 78.4966 |
| blendit_seg | DiffLoss | 20 | 60 | 1787 | 85 | 96.2663 | 91.1018 | 84.1074 |
| blendit_seg | MLP | 20 | 60 | 1787 | 50 | 96.4760 | 91.3191 | 84.4517 |
| fusion360seg | DiffLoss | 10 | 80 | 5366 | 65 | 72.7585 | 56.2762 | 43.0778 |
| fusion360seg | MLP | 10 | 80 | 5366 | 45 | 74.1274 | 57.4823 | 43.6564 |
| fusion360seg | DiffLoss | 20 | 160 | 5366 | 50 | 80.5242 | 64.2044 | 51.5944 |
| fusion360seg | MLP | 20 | 160 | 5366 | 75 | 80.9654 | 66.9286 | 53.8322 |
| mfcadpp_seg | DiffLoss | 10 | 249 | 8949 | 95 | 90.9131 | 84.1296 | 74.4113 |
| mfcadpp_seg | MLP | 10 | 249 | 8949 | 100 | 91.7675 | 86.0569 | 77.2072 |
| mfcadpp_seg | DiffLoss | 20 | 496 | 8949 | 100 | 93.7014 | 89.1176 | 81.4662 |
| mfcadpp_seg | MLP | 20 | 496 | 8949 | 75 | 93.7434 | 89.4415 | 82.1292 |
| solidletters_cls | DiffLoss | 10 | 260 | 19392 | 90 | 3.8624 | 2.2962 | 1.1769 |
| solidletters_cls | DiffLoss-xstart-epsilon | 10 | 260 | 19392 | 10 | 4.1099 | 0.3037 | 0.1581 |
| solidletters_cls | MLP | 10 | 260 | 19392 | 90 | 60.7828 | 60.4287 | 44.3527 |
| solidletters_cls | DiffLoss | 20 | 520 | 19392 | 100 | 9.6225 | 6.8842 | 3.7193 |
| solidletters_cls | DiffLoss-xstart-epsilon | 20 | 520 | 19392 | 100 | 15.3878 | 11.2583 | 6.7412 |
| solidletters_cls | MLP | 20 | 520 | 19392 | 100 | 76.8667 | 76.7628 | 62.7619 |
| tmcad_cls | DiffLoss | 10 | 100 | 1087 | 90 | 45.5382 | 46.0233 | 31.0365 |
| tmcad_cls | MLP | 10 | 100 | 1087 | 100 | 47.9301 | 47.6955 | 32.8600 |
| tmcad_cls | DiffLoss | 20 | 200 | 1087 | 75 | 57.4057 | 57.2585 | 41.3204 |
| tmcad_cls | MLP | 20 | 200 | 1087 | 40 | 59.3376 | 59.2533 | 43.5046 |

## Head comparison

Deltas are DiffLoss minus MLP in percentage points; positive values favor DiffLoss.

| Task | Shot | Accuracy delta | Macro F1 delta | Macro IoU delta |
|---|---:|---:|---:|---:|
| blendit_seg | 10 | -0.7924 | -2.9393 | -4.1466 |
| blendit_seg | 20 | -0.2097 | -0.2173 | -0.3443 |
| fusion360seg | 10 | -1.3689 | -1.2061 | -0.5785 |
| fusion360seg | 20 | -0.4412 | -2.7243 | -2.2378 |
| mfcadpp_seg | 10 | -0.8543 | -1.9273 | -2.7960 |
| mfcadpp_seg | 20 | -0.0420 | -0.3239 | -0.6630 |
| solidletters_cls | 10 | -56.9204 | -58.1325 | -43.1758 |
| solidletters_cls | 20 | -67.2442 | -69.8786 | -59.0427 |
| tmcad_cls | 10 | -2.3919 | -1.6722 | -1.8235 |
| tmcad_cls | 20 | -1.9319 | -1.9948 | -2.1842 |
