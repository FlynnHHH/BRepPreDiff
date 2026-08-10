# FabWave O_Rings / Washers 误检样本 ID

生成日期：2026-08-06

## 数据来源

- 模型：`runs/finetune/20260805-132209_fabwave_diffloss_20260805-unlabeled-v2`
- Checkpoint：`checkpoints/best.pt`（epoch 159）
- 测试集：`data/splits/fabwave_test_clean.txt`
- 推理结果已与该 run 的 `test_metrics.json` confusion matrix 完整核对一致。
- 相关误检共 35 个：`Washers → O_Rings` 26 个，`O_Rings → Washers` 9 个。

## Washers 被预测为 O_Rings（26 个）

| # | ID | 完整 sample ID |
|---:|---|---|
| 1 | `336b0daf-7954-4e8f-8860-b72b91b2235d` | `CAD16-24/Washers/STEP/336b0daf-7954-4e8f-8860-b72b91b2235d` |
| 2 | `361814c7-840f-4057-8a6f-34f8734c8895` | `CAD16-24/Washers/STEP/361814c7-840f-4057-8a6f-34f8734c8895` |
| 3 | `4744fb16-a342-4087-839f-eb13abac5521` | `CAD16-24/Washers/STEP/4744fb16-a342-4087-839f-eb13abac5521` |
| 4 | `509211d2-7082-4af1-8338-ed77a1fee5de` | `CAD16-24/Washers/STEP/509211d2-7082-4af1-8338-ed77a1fee5de` |
| 5 | `53795975-9086-4ba0-b48e-19073b6b2d95` | `CAD16-24/Washers/STEP/53795975-9086-4ba0-b48e-19073b6b2d95` |
| 6 | `55a7d754-0020-4618-b84f-58ff16db7a9b` | `CAD16-24/Washers/STEP/55a7d754-0020-4618-b84f-58ff16db7a9b` |
| 7 | `74dbc1b3-92d8-4fc0-89a1-d8df79b42296` | `CAD16-24/Washers/STEP/74dbc1b3-92d8-4fc0-89a1-d8df79b42296` |
| 8 | `75b6c6dd-7696-4aab-afc3-72ed90311480` | `CAD16-24/Washers/STEP/75b6c6dd-7696-4aab-afc3-72ed90311480` |
| 9 | `7c3ff4f1-063c-438f-a4d7-4519be5ec535` | `CAD16-24/Washers/STEP/7c3ff4f1-063c-438f-a4d7-4519be5ec535` |
| 10 | `83e98237-da68-4bb8-aca4-b252cb8eefa5` | `CAD16-24/Washers/STEP/83e98237-da68-4bb8-aca4-b252cb8eefa5` |
| 11 | `84214ad3-6631-4a6f-bc22-5d75ad1f446d` | `CAD16-24/Washers/STEP/84214ad3-6631-4a6f-bc22-5d75ad1f446d` |
| 12 | `88022793-95db-4be7-956f-8db20f624dbc` | `CAD16-24/Washers/STEP/88022793-95db-4be7-956f-8db20f624dbc` |
| 13 | `8c395ae8-943e-41fe-8835-ebbc5f5a6ba1` | `CAD16-24/Washers/STEP/8c395ae8-943e-41fe-8835-ebbc5f5a6ba1` |
| 14 | `986dcf3b-f516-4f2a-b84d-e651c95a3d0d` | `CAD16-24/Washers/STEP/986dcf3b-f516-4f2a-b84d-e651c95a3d0d` |
| 15 | `9b698bac-5835-4540-b9d0-43eef29dd296` | `CAD16-24/Washers/STEP/9b698bac-5835-4540-b9d0-43eef29dd296` |
| 16 | `9e443494-58a6-4bf0-8885-2e00cb5fa611` | `CAD16-24/Washers/STEP/9e443494-58a6-4bf0-8885-2e00cb5fa611` |
| 17 | `9f78b434-1c3d-4d14-b431-29acb3c8140c` | `CAD16-24/Washers/STEP/9f78b434-1c3d-4d14-b431-29acb3c8140c` |
| 18 | `a95f7ae7-56c5-4068-b301-7052026486ef` | `CAD16-24/Washers/STEP/a95f7ae7-56c5-4068-b301-7052026486ef` |
| 19 | `b48e27bf-f583-4c0a-98b0-88e9b212d85d` | `CAD16-24/Washers/STEP/b48e27bf-f583-4c0a-98b0-88e9b212d85d` |
| 20 | `bc01fce1-63fe-495e-9cdc-4e8d6e663510` | `CAD16-24/Washers/STEP/bc01fce1-63fe-495e-9cdc-4e8d6e663510` |
| 21 | `c53072c0-0617-417d-afa2-ff4ec8235d95` | `CAD16-24/Washers/STEP/c53072c0-0617-417d-afa2-ff4ec8235d95` |
| 22 | `c5f702a5-e2bf-4988-8e2f-173ae5f77584` | `CAD16-24/Washers/STEP/c5f702a5-e2bf-4988-8e2f-173ae5f77584` |
| 23 | `c75b68de-9040-4745-acb6-7a533416695e` | `CAD16-24/Washers/STEP/c75b68de-9040-4745-acb6-7a533416695e` |
| 24 | `cc342b73-1f33-4fb0-a3d6-c188adcce554` | `CAD16-24/Washers/STEP/cc342b73-1f33-4fb0-a3d6-c188adcce554` |
| 25 | `ee29271e-b7c1-4d18-9aa0-0b74d88bf9c6` | `CAD16-24/Washers/STEP/ee29271e-b7c1-4d18-9aa0-0b74d88bf9c6` |
| 26 | `f6defc2a-c913-498f-9612-0cabf69edbe3` | `CAD16-24/Washers/STEP/f6defc2a-c913-498f-9612-0cabf69edbe3` |

## O_Rings 被预测为 Washers（9 个）

| # | ID | 完整 sample ID |
|---:|---|---|
| 1 | `16877b0a-41e9-490d-ad13-a850ba1d8caa` | `CAD_1_15_Classes/O_Rings/STEP/16877b0a-41e9-490d-ad13-a850ba1d8caa` |
| 2 | `1cc503b5-4017-44cb-8f67-7d23160e723c` | `CAD_1_15_Classes/O_Rings/STEP/1cc503b5-4017-44cb-8f67-7d23160e723c` |
| 3 | `2ded5cf2-02f9-4c62-9ff9-f7bcd1cd2a1a` | `CAD_1_15_Classes/O_Rings/STEP/2ded5cf2-02f9-4c62-9ff9-f7bcd1cd2a1a` |
| 4 | `5d83fa0a-e0e6-4e0b-90ef-91ba55ed797b` | `CAD_1_15_Classes/O_Rings/STEP/5d83fa0a-e0e6-4e0b-90ef-91ba55ed797b` |
| 5 | `70b1e3d7-cbf0-44d6-824a-5b35d32ad17f` | `CAD_1_15_Classes/O_Rings/STEP/70b1e3d7-cbf0-44d6-824a-5b35d32ad17f` |
| 6 | `9ba4ef4b-7e9a-4526-99e1-4be9adaf5729` | `CAD_1_15_Classes/O_Rings/STEP/9ba4ef4b-7e9a-4526-99e1-4be9adaf5729` |
| 7 | `b35bc052-df5d-490e-8c16-72e4788c76fb` | `CAD_1_15_Classes/O_Rings/STEP/b35bc052-df5d-490e-8c16-72e4788c76fb` |
| 8 | `d4c16164-8456-4599-b7b4-fd8bbedfbc89` | `CAD_1_15_Classes/O_Rings/STEP/d4c16164-8456-4599-b7b4-fd8bbedfbc89` |
| 9 | `dce8c0cf-bcf5-45a8-a4c9-0280de5b365b` | `CAD_1_15_Classes/O_Rings/STEP/dce8c0cf-bcf5-45a8-a4c9-0280de5b365b` |
