# Tracking Metrics — All Models, All Epochs

> **Evaluated:** April 9, 2026  
> **Dataset:** P-DESTRE (aerial pedestrian tracking)  
> **Metric computation:** motmetrics library, IoU threshold = 0.5, per-sequence mean ± std  
> **Selection criterion:** Best epoch by MOTA (standard practice in MOT literature)

---

## Model Overview

| # | Model | Backbone | Concepts | Learnable Weights | Val Split | Epochs Available | Status |
|---|-------|----------|----------|-------------------|-----------|-----------------|--------|
| 1 | R50 2-Concept | ResNet-50 + Deformable DETR | 2 (gender, upper_body) | No | val_0 (14 seqs) | 0–9 | **Complete** |
| 2 | R50 3-Concept | ResNet-50 + Deformable DETR | 3 (gender, upper_body, lower_body) | No | val_0 (14 seqs) | 0–2 | Still training |
| 3 | R50 7c Learnable v2 | ResNet-50 + Deformable DETR | 7 (all) | **Yes** | val_1 (15 seqs) | 3–8 | Complete (no epoch 9) |
| 4 | R50 7c Learnable Fold 1 | ResNet-50 + Deformable DETR | 7 (all) | **Yes** | val_1 (15 seqs) | 9 only | Complete (only last epoch eval'd) |
| 5 | RF-DETR 2-Concept | DINOv2-Large (ViT-L) | 2 (gender, upper_body) | No | val_0 (14 seqs) | 0–9 | **Complete** |
| 6 | RF-DETR 7c **NO** Learnable Weights | DINOv2-Large (ViT-L) | 7 (all) | **No** (despite "learnable" in dir name) | val_0 (14 seqs) | 0–5 | Still training |
| 7 | RF-DETR 7c **WITH** Learnable Weights | DINOv2-Large (ViT-L) | 7 (all) | **Yes** (dir name has "lw") | val_0 (14 seqs) | 0–2 | Still training |
| 8 | R50 SAM-MOTIP (DanceTrack) | ResNet-50 + Deformable DETR | SAM masks | N/A (unsupervised) | DanceTrack val (25 seqs) | 0–6 | Still training (epoch 7) |
| 9 | RF-DETR Base SAM-MOTIP (DanceTrack) | DINOv2 Windowed Small + RF-DETR Base | SAM masks | N/A (unsupervised) | DanceTrack val (25 seqs) | 0–9 | **Complete** |

### Important naming clarification
- `rfdetr_large_motip_pdestre_7concepts_learnable_fold0` → **Does NOT use learnable weights** despite the name. This was the first 7-concept RF-DETR run with manually-set/equal concept weights.
- `rfdetr_large_motip_pdestre_7concepts_lw_fold0` → **DOES use learnable weights**. This is the newer run with `USE_LEARNABLE_TASK_WEIGHTS: True`.

---

## 1. R50 2-Concept (Fold 0)

**Config:** ResNet-50 + Deformable DETR, 2 concepts (gender, upper_body), val_0 (14 seqs, ~224K GT)  
**Best epoch: 7** (MOTA = 49.46%)

| Epoch | MOTA (%) | IDF1 (%) | Precision (%) | Recall (%) | TP | FP | FN | ID Sw | Detections |
|------:|---------:|---------:|--------------:|-----------:|-------:|------:|-------:|------:|-----------:|
| 0 | 41.03 ± 7.07 | 37.03 ± 7.53 | 83.22 | 55.33 | 124,253 | 24,647 | 99,802 | 6,491 | 148,900 |
| 1 | 42.04 ± 7.46 | 38.65 ± 9.81 | 80.57 | 59.20 | 126,769 | 28,825 | 97,286 | 5,843 | 155,594 |
| 2 | 45.57 ± 9.87 | 47.28 ± 11.77 | 75.86 | 73.94 | 161,591 | 50,507 | 62,464 | 10,691 | 212,098 |
| 3 | 43.13 ± 8.60 | 44.41 ± 10.84 | 75.75 | 68.28 | 150,176 | 45,283 | 73,879 | 7,240 | 195,459 |
| 4 | 46.51 ± 9.86 | 45.86 ± 14.57 | 80.35 | 67.65 | 147,151 | 34,796 | 76,904 | 11,997 | 181,947 |
| 5 | 42.60 ± 7.59 | 38.83 ± 9.26 | 82.12 | 57.42 | 127,686 | 26,432 | 96,369 | 4,721 | 154,118 |
| 6 | 47.57 ± 8.25 | 45.90 ± 10.68 | 81.34 | 65.12 | 143,165 | 30,812 | 80,890 | 6,162 | 173,977 |
| **7** | **49.46 ± 8.02** | **48.57 ± 10.51** | **82.04** | **66.21** | **144,508** | **30,145** | **79,547** | **4,583** | **174,653** |
| 8 | 46.70 ± 7.72 | 48.08 ± 12.60 | 81.64 | 64.26 | 140,031 | 29,118 | 84,024 | 7,571 | 169,149 |
| 9 | 48.97 ± 7.05 | 50.57 ± 10.44 | 81.99 | 65.17 | 143,292 | 28,990 | 80,763 | 3,560 | 172,282 |

**Observations:**
- Stable training with gradual improvement. Peak MOTA at epoch 7, peak IDF1 at epoch 9.
- Epoch 2 is interesting: highest recall (73.94%) but lowest precision (75.86%) — the model was aggressive with detections.
- Minimal overfitting — epochs 7–9 all perform similarly well (MOTA 47–49%).
- ID switches lowest at epoch 9 (3,560) suggesting the ID model continues to improve even when MOTA plateaus.

---

## 2. R50 3-Concept (Fold 0)

**Config:** ResNet-50 + Deformable DETR, 3 concepts (gender, upper_body, lower_body), val_0 (14 seqs, ~224K GT)  
**Best epoch: 2** (MOTA = 50.20%) — Still training, only 3 epochs evaluated so far.

| Epoch | MOTA (%) | IDF1 (%) | Precision (%) | Recall (%) | TP | FP | FN | ID Sw | Detections |
|------:|---------:|---------:|--------------:|-----------:|-------:|------:|-------:|------:|-----------:|
| 0 | 38.72 ± 10.93 | 36.01 ± 12.26 | 78.77 | 58.75 | 126,423 | 31,901 | 97,632 | 8,196 | 158,324 |
| 1 | 45.67 ± 10.89 | 38.61 ± 14.79 | 81.88 | 62.43 | 133,510 | 29,158 | 90,545 | 6,226 | 162,668 |
| **2** | **50.20 ± 7.60** | **48.09 ± 14.60** | **78.67** | **72.65** | **156,164** | **39,286** | **67,891** | **5,010** | **195,450** |

**Observations:**
- Strong upward trajectory — MOTA improved by +11.5 pts in just 3 epochs.
- Epoch 2 already exceeds R50 2-Concept's best (50.20% vs 49.46%), suggesting the 3rd concept (lower_body) provides additional discriminative value.
- Recall jumped significantly in epoch 2 (72.65% — highest of any R50 model at any epoch), though precision dipped to 78.67%.
- High variance (±7.60 to ±14.60) suggests some validation sequences are much harder than others.

---

## 3. R50 7-Concept Learnable v2 (Fold 0)

**Config:** ResNet-50 + Deformable DETR, 7 concepts (all: gender, hairstyle, head_acc, upper_body, lower_body, feet, accessories), **learnable task weights**, val_1 (15 seqs, ~300K GT)  
**Best epoch: 3** (MOTA = 49.62%) — Severe overfitting after epoch 3.

| Epoch | MOTA (%) | IDF1 (%) | Precision (%) | Recall (%) | TP | FP | FN | ID Sw | Detections |
|------:|---------:|---------:|--------------:|-----------:|-------:|------:|-------:|------:|-----------:|
| **3** | **49.62 ± 10.20** | **45.12 ± 10.98** | **85.33** | **61.35** | **187,710** | **32,949** | **112,315** | **4,087** | **220,659** |
| 4 | 34.06 ± 11.58 | 29.99 ± 11.09 | 86.29 | 41.79 | 120,850 | 18,418 | 179,175 | 3,524 | 139,268 |
| 5 | 33.10 ± 10.30 | 32.57 ± 10.69 | 82.88 | 42.97 | 121,841 | 26,431 | 178,184 | 3,145 | 148,272 |
| 6 | 39.88 ± 14.11 | 34.67 ± 13.88 | 87.08 | 47.70 | 135,106 | 20,781 | 164,919 | 2,725 | 155,887 |
| 7 | 37.10 ± 13.53 | 33.19 ± 12.37 | 87.83 | 43.97 | 129,070 | 18,271 | 170,955 | 2,642 | 147,341 |
| 8 | 35.54 ± 14.61 | 32.85 ± 12.63 | 89.68 | 40.85 | 117,678 | 12,740 | 182,347 | 2,359 | 130,418 |

**Observations:**
- Catastrophic overfitting after epoch 3: MOTA drops from 49.62% to ~34% by epoch 4, never recovers.
- Precision *increases* as epochs progress (85→90%) but recall *plummets* (61%→41%), meaning the model becomes increasingly conservative/sparse.
- Detections drop from 220K to 130K — the model learns to predict fewer and fewer tracks.
- Note: val_1 has 300K GT objects (vs 224K for val_0), so direct comparison to val_0 models requires care.

---

## 4. R50 7-Concept Learnable (Fold 1)

**Config:** ResNet-50 + Deformable DETR, 7 concepts, learnable task weights, val_1 (15 seqs, ~300K GT)  
**Only epoch 9 evaluated.** This was an earlier/different training run than v2.

| Epoch | MOTA (%) | IDF1 (%) | Precision (%) | Recall (%) | TP | FP | FN | ID Sw | Detections |
|------:|---------:|---------:|--------------:|-----------:|-------:|------:|-------:|------:|-----------:|
| 9 | 10.91 ± 13.72 | 12.24 ± 13.26 | N/A | 14.77 | 34,175 | 6,974 | 265,850 | 1,398 | 41,149 |

**Observations:**
- Very poor performance. Only 41K detections out of 300K GT — model drastically under-detects.
- Very high variance (±13.72%) suggests near-random performance on some sequences.
- This run likely suffered from different hyperparameters or training issues compared to v2.
- Precision is NaN because some sequences have 0 predictions, causing division by zero.

---

## 5. RF-DETR Large 2-Concept (Fold 0)

**Config:** DINOv2-Large (ViT-L) + RF-DETR, 2 concepts (gender, upper_body), val_0 (14 seqs, ~224K GT)  
**Best epoch: 0** (MOTA = 32.72%) — **Severe overfitting from epoch 1 onward.**

| Epoch | MOTA (%) | IDF1 (%) | Precision (%) | Recall (%) | TP | FP | FN | ID Sw | Detections |
|------:|---------:|---------:|--------------:|-----------:|-------:|------:|-------:|------:|-----------:|
| **0** | **32.72 ± 9.03** | **33.28 ± 11.34** | **84.29** | **42.90** | **88,936** | **15,872** | **135,119** | **4,194** | **104,808** |
| 1 | 18.62 ± 5.85 | 23.27 ± 6.62 | 77.63 | 28.74 | 62,500 | 16,150 | 161,555 | 3,689 | 78,650 |
| 2 | 15.27 ± 6.31 | 18.27 ± 7.95 | 76.31 | 24.22 | 52,890 | 14,277 | 171,165 | 3,465 | 67,167 |
| 3 | 14.30 ± 6.25 | 18.39 ± 8.15 | 76.19 | 23.02 | 50,407 | 13,612 | 173,648 | 3,135 | 64,019 |
| 4 | 13.59 ± 6.37 | 17.99 ± 8.25 | 74.88 | 23.45 | 50,429 | 15,058 | 173,626 | 3,813 | 65,487 |
| 5 | 14.73 ± 7.64 | 20.39 ± 7.52 | 72.52 | 27.29 | 61,681 | 21,773 | 162,374 | 4,496 | 83,454 |
| 6 | 7.26 ± 5.06 | 11.59 ± 4.85 | 71.22 | 14.69 | 33,862 | 12,537 | 190,193 | 3,155 | 46,399 |
| 7 | 8.28 ± 5.39 | 12.09 ± 5.80 | 73.57 | 15.03 | 34,265 | 11,165 | 189,790 | 2,930 | 45,430 |
| 8 | 5.53 ± 3.69 | 8.40 ± 3.94 | 70.76 | 11.77 | 26,933 | 10,535 | 197,122 | 3,047 | 37,468 |
| 9 | 5.98 ± 4.12 | 9.07 ± 4.42 | 71.50 | 11.84 | 27,253 | 10,227 | 196,802 | 2,759 | 37,480 |

**Observations:**
- Dramatic overfitting: MOTA drops from 32.72% (epoch 0) to 5.98% (epoch 9) — a **27-point decline**.
- Recall collapses from 42.90% → 11.84% while precision also degrades (84% → 71%).
- Detections drop from 105K to 37K — the model essentially stops producing useful tracks.
- This suggests the DINOv2-Large backbone + RF-DETR architecture struggles with the P-DESTRE dataset size. The large model capacity overfits quickly to the small training set (44 sequences).

---

## 6. RF-DETR 7-Concept — NO Learnable Weights (Fold 0)

**Config:** DINOv2-Large (ViT-L) + RF-DETR, 7 concepts (all), **equal/manual weights** (NOT learnable despite directory name), val_0 (14 seqs, ~224K GT)  
**Best epoch: 0** (MOTA = 41.24%) — Still training, 6 epochs evaluated.

> **Note:** The directory name `rfdetr_large_motip_pdestre_7concepts_learnable_fold0` is misleading. This model uses fixed concept weights, not learnable ones.

| Epoch | MOTA (%) | IDF1 (%) | Precision (%) | Recall (%) | TP | FP | FN | ID Sw | Detections |
|------:|---------:|---------:|--------------:|-----------:|-------:|------:|-------:|------:|-----------:|
| **0** | **41.24 ± 7.87** | **49.73 ± 12.18** | 75.86 | **67.16** | **141,780** | **43,552** | **82,275** | 8,363 | **185,332** |
| 1 | 21.89 ± 6.98 | 28.00 ± 4.80 | 73.98 | 38.64 | 83,999 | 25,310 | 140,056 | 4,878 | 109,309 |
| 2 | 26.03 ± 5.18 | 32.00 ± 8.69 | 75.03 | 43.33 | 93,616 | 28,724 | 130,439 | 5,690 | 122,340 |
| 3 | 17.49 ± 4.26 | 19.81 ± 5.03 | **78.28** | 28.13 | 61,056 | 15,303 | 162,999 | 5,173 | 76,359 |
| 4 | 19.48 ± 7.10 | 23.63 ± 9.08 | 76.37 | 32.36 | 68,765 | 19,693 | 155,290 | 5,246 | 88,458 |
| 5 | 17.69 ± 7.58 | 27.44 ± 8.87 | 69.69 | 39.09 | 83,881 | 32,073 | 140,174 | 6,810 | 115,954 |

**Observations:**
- Same overfitting pattern as RF-DETR 2-Concept but less severe (MOTA 41% → 18% vs 33% → 6%).
- Epoch 0 is the standout: highest IDF1 (49.73%) of any RF-DETR model — 7 concepts clearly help identity association.
- Despite overfitting on MOTA, the model retains some capability (MOTA doesn't collapse to single digits).
- High ID switches at epoch 0 (8,363) — the model is aggressive (185K detections) but also confused about identities.

---

## 7. RF-DETR 7-Concept — WITH Learnable Weights (Fold 0)

**Config:** DINOv2-Large (ViT-L) + RF-DETR, 7 concepts (all), **learnable task weights** (`USE_LEARNABLE_TASK_WEIGHTS: True`), val_0 (14 seqs, ~224K GT)  
**Best epoch: 2** (MOTA = 15.35%) — Still training, only 3 epochs evaluated.

> **Note:** This is the newer run (`rfdetr_large_motip_pdestre_7concepts_lw_fold0`) that actually uses learnable weights. It's still very early in training.

| Epoch | MOTA (%) | IDF1 (%) | Precision (%) | Recall (%) | TP | FP | FN | ID Sw | Detections |
|------:|---------:|---------:|--------------:|-----------:|-------:|------:|-------:|------:|-----------:|
| 0 | 14.62 ± 6.13 | 19.65 ± 8.14 | 73.49 | 25.76 | 54,349 | 20,675 | 169,706 | 1,455 | 75,024 |
| 1 | 10.05 ± 6.70 | 20.60 ± 7.85 | 64.19 | 28.11 | 61,956 | 31,663 | 162,099 | 5,414 | 93,619 |
| **2** | **15.35 ± 5.41** | **25.93 ± 7.87** | **65.67** | **39.20** | **85,150** | **41,815** | **138,905** | **6,033** | **126,965** |

**Observations:**
- Starting from a much lower baseline than the non-learnable version (14.62% vs 41.24% at epoch 0).
- The model is slowly improving — recall increasing (26% → 39%), IDF1 climbing (20% → 26%).
- Detections are ramping up (75K → 127K) suggesting the model is learning to detect more objects.
- Unlike the non-learnable version, there's **no overfitting yet** — metrics are still trending upward.
- Low precision (65–73%) and high FP count means the model is being aggressive but inaccurate so far.
- Too early to judge final performance — needs more epochs to converge.

---

## 8. R50 SAM-MOTIP (DanceTrack)

**Config:** ResNet-50 + Deformable DETR, SAM ViT-B concept masks via masked average pooling + MLP fusion, DanceTrack val (25 seqs, 225,148 GT)
**Best epoch: 6** (HOTA = 59.89%) — Still training, 7 epochs evaluated (0–6).

| Epoch | HOTA (%) | DetA (%) | AssA (%) | DetPr (%) | DetRe (%) | AssPr (%) | AssRe (%) | MOTA (%) | IDF1 (%) | TP | FN | FP | IDSW |
|------:|---------:|---------:|---------:|----------:|----------:|----------:|----------:|---------:|---------:|-------:|------:|------:|------:|
| 0 | 43.57 | 70.08 | 27.23 | 83.53 | 77.54 | 56.36 | 33.06 | 72.93 | 42.88 | 197,687 | 27,461 | 11,317 | 22,176 |
| 1 | 47.85 | 71.06 | 32.44 | 82.89 | 79.24 | 55.34 | 39.71 | 79.14 | 49.45 | 201,356 | 23,792 | 13,897 | 9,288 |
| 2 | 52.81 | 71.96 | 39.00 | 83.73 | 79.98 | 58.96 | 46.69 | 80.85 | 55.88 | 202,338 | 22,810 | 12,727 | 7,588 |
| 3 | 53.49 | 72.71 | 39.58 | 82.92 | 81.76 | 59.00 | 47.47 | 81.36 | 56.42 | 205,639 | 19,509 | 16,362 | 6,096 |
| 4 | 53.45 | 73.51 | 39.09 | 85.06 | 80.74 | 58.71 | 46.89 | 82.87 | 56.88 | 203,183 | 21,965 | 10,544 | 6,056 |
| 5 | 54.98 | 71.95 | 42.22 | 83.05 | 80.89 | 62.73 | 49.53 | 80.19 | 58.30 | 202,925 | 22,223 | 16,381 | 6,002 |
| **6** | **59.89** | **74.55** | **48.36** | **85.11** | **82.39** | **65.45** | **55.74** | **83.93** | **63.20** | **205,506** | **19,642** | **12,444** | **4,101** |

**Observations:**
- Steady, monotonic improvement in HOTA over all 7 epochs with no overfitting — a stark contrast to RF-DETR models on P-DESTRE.
- AssA (identity association) is the primary driver of improvement: +21.1 pts (27.2% → 48.4%) vs DetA's +4.5 pts (70.1% → 74.5%).
- IDSW drops from 22,176 at epoch 0 to 4,101 at epoch 6 — the concept fusion network learns to produce identity-consistent features.
- Strong LR-decay boost at epoch 6 (milestone): HOTA jumps +4.9 pts over epoch 5.
- Training losses: total loss decreases from 4.96 (epoch 0) to 2.10 (epoch 6); DETR loss 3.31→1.80, ID loss 1.65→0.30.

---

## 9. RF-DETR Base SAM-MOTIP (DanceTrack)

**Config:** RF-DETR Base (DINOv2 Windowed Small encoder), SAM ViT-B concept masks via masked average pooling + MLP fusion, batch_size=1, accumulate_steps=2, resolution=560, DanceTrack val (25 seqs, 225,148 GT)
**Best epoch: 9** (HOTA = 38.91%) — Training complete, all 10 epochs evaluated.

| Epoch | HOTA (%) | DetA (%) | AssA (%) | DetPr (%) | DetRe (%) | AssPr (%) | AssRe (%) | MOTA (%) | IDF1 (%) | TP | FN | FP | IDSW |
|------:|---------:|---------:|---------:|----------:|----------:|----------:|----------:|---------:|---------:|-------:|------:|------:|------:|
| 0 | 22.95 | 43.54 | 12.66 | 56.71 | 49.98 | 49.22 | 16.70 | 24.03 | 24.48 | 148,698 | 76,450 | 49,731 | 44,869 |
| 1 | 26.56 | 49.50 | 14.62 | 63.75 | 56.30 | 47.36 | 19.03 | 48.07 | 30.04 | 172,531 | 52,617 | 26,285 | 38,025 |
| 2 | 29.96 | 48.98 | 18.64 | 66.45 | 54.71 | 45.94 | 23.43 | 54.26 | 35.39 | 164,300 | 60,848 | 21,096 | 21,048 |
| 3 | 31.56 | 53.71 | 18.67 | 69.84 | 60.26 | 48.29 | 24.12 | 57.17 | 36.32 | 177,061 | 48,087 | 17,198 | 31,143 |
| 4 | 33.72 | 59.51 | 19.29 | 74.09 | 66.37 | 49.77 | 24.25 | 63.81 | 36.56 | 187,847 | 37,301 | 13,827 | 30,345 |
| 5 | 29.71 | 54.63 | 16.43 | 70.13 | 61.38 | 52.91 | 20.94 | 49.40 | 33.00 | 178,876 | 46,272 | 18,188 | 49,457 |
| 6 | 36.91 | 60.14 | 22.88 | 74.32 | 66.79 | 51.78 | 29.60 | 62.68 | 41.97 | 188,905 | 36,243 | 13,421 | 34,353 |
| 7 | 38.71 | 61.06 | 24.79 | 74.75 | 67.68 | 54.83 | 31.06 | 63.56 | 43.38 | 191,474 | 33,674 | 12,396 | 35,971 |
| 8 | 37.64 | 61.53 | 23.27 | 75.50 | 67.85 | 56.49 | 29.57 | 60.55 | 41.35 | 191,347 | 33,801 | 10,967 | 44,053 |
| **9** | **38.91** | **62.31** | **24.56** | **75.61** | **68.81** | **54.66** | **30.97** | **64.53** | **42.51** | **193,280** | **31,868** | **11,626** | **36,363** |

**Observations:**
- Overall improving trend but highly non-monotonic: epoch 5 shows a sharp regression (HOTA drops −4.0 pts from epoch 4), recovering by epoch 6.
- Detection quality improves substantially: DetA goes from 43.54% to 62.31% (+18.8 pts), DetPr from 56.71% to 75.61%.
- Association quality remains the bottleneck: AssA reaches only 24.56% at best vs R50 SAM's 48.36% — roughly half.
- Extremely high IDSW throughout (30K–49K per epoch). Even at best epoch, 36,363 identity switches vs R50 SAM's 4,101. This is 8.9× worse.
- The epoch-5 regression (MOTA drops from 63.81% to 49.40%) coincides with the highest IDSW count (49,457), suggesting the model temporarily destabilized before the LR reduction at epoch 6.
- Training losses: total loss 9.77 (epoch 0) → 4.56 (epoch 9); DETR loss 7.81→3.74, ID loss 1.96→0.82 — losses continue decreasing but tracking quality plateaus.
- **Conclusion:** RF-DETR Base with SAM concept fusion significantly underperforms the R50 variant on DanceTrack. The DINOv2 Windowed Small encoder's features do not integrate as effectively with the concept fusion module.

---

## Cross-Model Comparison (Best Epoch)

### ResNet-50 Backbone Models

| Model | Best Epoch | MOTA (%) | IDF1 (%) | Precision (%) | Recall (%) | Detections | ID Sw |
|-------|----------:|---------:|---------:|--------------:|-----------:|-----------:|------:|
| **R50 3-Concept** | 2 | **50.20** | 48.09 | 78.67 | **72.65** | 195,450 | 5,010 |
| R50 7c Learnable v2 | 3 | 49.62 | 45.12 | **85.33** | 61.35 | 220,659 | **4,087** |
| R50 2-Concept | 7 | 49.46 | **48.57** | 82.04 | 66.21 | 174,653 | 4,583 |
| R50 7c Learnable Fold 1 | 9 | 10.91 | 12.24 | N/A | 14.77 | 41,149 | 1,398 |

> **Note:** R50 3-Concept has only 3 epochs so far — may improve or overfit with more training. R50 7c Learnable v2 uses val_1 (15 seqs, 300K GT) while the others use val_0 (14 seqs, 224K GT), so direct comparison requires caution.

### DINOv2/RF-DETR Backbone Models

| Model | Best Epoch | MOTA (%) | IDF1 (%) | Precision (%) | Recall (%) | Detections | ID Sw |
|-------|----------:|---------:|---------:|--------------:|-----------:|-----------:|------:|
| **RF-DETR 7c NO Learnable** | 0 | **41.24** | **49.73** | 75.86 | **67.16** | **185,332** | 8,363 |
| RF-DETR 2-Concept | 0 | 32.72 | 33.28 | **84.29** | 42.90 | 104,808 | 4,194 |
| RF-DETR 7c WITH Learnable | 2 | 15.35 | 25.93 | 65.67 | 39.20 | 126,965 | 6,033 |

### Overall Best (by MOTA)

| Rank | Model | Best Epoch | MOTA | IDF1 |
|-----:|-------|----------:|-----:|-----:|
| 1 | R50 3-Concept (still training) | 2 | **50.20%** | 48.09% |
| 2 | R50 7c Learnable v2 | 3 | 49.62% | 45.12% |
| 3 | R50 2-Concept | 7 | 49.46% | 48.57% |
| 4 | RF-DETR 7c NO Learnable | 0 | 41.24% | **49.73%** |
| 5 | RF-DETR 2-Concept | 0 | 32.72% | 33.28% |
| 6 | RF-DETR 7c WITH Learnable | 2 | 15.35% | 25.93% |
| 7 | R50 7c Learnable Fold 1 | 9 | 10.91% | 12.24% |

### DanceTrack SAM-MOTIP Models (by HOTA)

| Rank | Model | Best Epoch | HOTA | MOTA | IDF1 | AssA |
|-----:|-------|----------:|-----:|-----:|-----:|-----:|
| 1 | **R50 SAM-MOTIP** | 6 | **59.89%** | **83.93%** | **63.20%** | **48.36%** |
| 2 | RF-DETR Base SAM-MOTIP | 9 | 38.91% | 64.53% | 42.51% | 24.56% |

---

## Key Findings

### 1. ResNet-50 consistently outperforms RF-DETR on P-DESTRE
- Best R50 model achieves MOTA 50.20% vs RF-DETR's 41.24% — despite DINOv2 being a far larger backbone
- R50 models have COCO-pretrained DETR weights, while RF-DETR trains from scratch → the smaller dataset (44 training sequences) benefits more from transfer learning

### 2. All RF-DETR models overfit catastrophically on P-DESTRE
- Best performance is always at epoch 0 (or very early)
- Performance degrades monotonically: RF-DETR 2c drops from 32.72% to 5.98% over 10 epochs
- The DINOv2-Large backbone (304M params) has enormous capacity that overfits on the small P-DESTRE training set

### 3. More concepts help — up to a point
- R50: 3c (50.20%) ≈ 7c learnable (49.62%) > 2c (49.46%) — though 3c is still training
- RF-DETR: 7c no-learnable (41.24%) >> 2c (32.72%) — 7 concepts provide much more identity information for the larger backbone
- R50 7c Fold 1 (10.91%) shows that 7 concepts can also fail badly with wrong hyperparameters

### 4. Learnable weights slow RF-DETR convergence but may prevent overfitting
- RF-DETR 7c WITH learnable weights starts much lower (14.62% epoch 0 vs 41.24%) but is still improving at epoch 2 with no overfitting
- The non-learnable version peaks at epoch 0 then immediately degrades
- Needs more epochs to see if learnable weights eventually reach or surpass the non-learnable peak

### 5. Overfitting timeline differs by architecture
- R50 2-Concept: Gradual improvement, peaks at epoch 7, minimal degradation
- R50 7c Learnable v2: Sharp peak at epoch 3, then sudden collapse
- RF-DETR (all): Peaks at epoch 0, immediate decline

### 6. R50 SAM-MOTIP achieves the best overall tracking performance
- R50 SAM-MOTIP on DanceTrack: HOTA=59.89%, MOTA=83.93%, IDF1=63.20% (epoch 6 of 7 completed)
- Dramatically outperforms all P-DESTRE models (which peak at ~50% MOTA) — though on a different dataset
- SAM concept fusion with ResNet-50 shows no overfitting through 7 epochs, with consistently improving metrics
- The concept fusion module learns to extract identity-relevant features from SAM masks: AssA improves from 27.2% to 48.4%

### 7. RF-DETR Base SAM-MOTIP significantly underperforms R50 on DanceTrack
- RF-DETR Base SAM: HOTA=38.91%, MOTA=64.53% (epoch 9) — less than R50 SAM's epoch 0 (HOTA=43.57%)
- The DINOv2 Windowed Small encoder features do not integrate well with the SAM concept fusion module
- Identity switches are 8.9× higher than R50 SAM (36,363 vs 4,101), indicating severe re-ID failure
- Non-monotonic training with sharp regressions (epoch 5 dip) suggests training instability

### 8. SAM-based concept fusion is dataset-appropriate
- DanceTrack features dancers in similar costumes → attribute-based concepts (gender, clothing) would be uninformative
- SAM masks provide foreground segmentation that captures body shape, pose, and texture differences
- The R50 SAM model proves that unsupervised concept features can outperform supervised attribute features when the attribute space is not discriminative for the target domain

---

## Raw Data Reference

Results saved to `evaluation_results_attached.json` with full per-epoch breakdowns including standard deviations for all metrics.
