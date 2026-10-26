# SCCR: Support-Constrained Coordinate Refinement

Minimal research materials by **Jin Huang** (first/corresponding author, Guangxi Vocational Normal University) and **Qisen Gao** (Gansu University of Political Science and Law).

This is a controlled study of an unrestricted dynamic extension of RankSEG-RMA, not a claim that the official decoder exhibits class activation explosion. SCCR restores the original probability-defined destination support. The manuscript is prepared for Image and Vision Computing; it has not been submitted or accepted.

## Quick checks (no dataset, weights, training or network access)

Use Python 3.12 and NumPy (the recorded version is in requirements.txt):

```shell
python -m pip install -r requirements.txt
python verify_math.py
python reproduce_tables.py
```

The first checks exact coordinate gains against full objective recomputation on 40 small problems and verifies feasibility and nondecreasing surrogate values. The second independently derives per-image metrics from 80 saved confusion matrices and reproduces reported means and paired bootstrap intervals (5,000 draws; seed 20261023). It checks every result against the frozen tables. This is numerical result verification, not rerunning the segmentation network.

| Method | GT-present Dice (%) | Union-present Dice (%) | False-only classes/image |
|---|---:|---:|---:|
| Argmax | 78.43 | 69.99 | 0.49 |
| Original RMA | 80.29 | 55.21 | 1.64 |
| Unrestricted dynamic | 80.48 | 10.81 | 17.09 |
| SCCR | 80.76 | 55.83 | 1.60 |

SCCR minus RMA GT-present Dice is +0.475 percentage points, 95% CI [-0.240, 1.156]. This does not establish primary-metric superiority. Only one backbone, one threshold and one scan were tested. A class suppressed by the fixed gate cannot be recovered through the restricted overlap updates.

## Optional full Confirm B inference

1. Reuse the official VOC2012 train/validation tar archive and LRASPP-MobileNetV3-Large COCO_WITH_VOC_LABELS_V1 checkpoint. Verify their hashes in input_checksums.json. No automatic downloads occur.
2. Obtain `exp/rankseg_rma.py` from [the upstream repository](https://github.com/ZixunWang/RankSEG-RMA) at commit `205c71c3c5ff1cfd71fe2173c34bcb0211a0792d`; its expected SHA-256 is recorded. Do not change the decoder.
3. Install the recorded PyTorch/torchvision/Pillow environment from requirements-inference.txt using an appropriate official wheel index. The original installed wheels were torch 2.13.0+cu130 and torchvision 0.28.0+cu130, used on CPU. Other builds may not be bitwise identical.
4. Run with explicit input paths (Windows example):

```powershell
python run_voc.py --archive E:/Datasets/PASCAL2012/VOC2012/VOC2012_VOCtrainval_11-May-2012.tar --weights E:/Datasets/pretrained_model_cache/torch/hub/checkpoints/lraspp_mobilenet_v3_large-d234d4ea.pth --official-code /path/to/RankSEG-RMA/exp/rankseg_rma.py --output outputs/confirm_b
python reproduce_tables.py --results outputs/confirm_b
```

The frozen split is official validation positions 101–180 (one-based); exact IDs and image/mask hashes are retained. Inference is CPU FP32 with two threads, float64 coordinate statistics, row-major order, one pass and tolerance 1e-12. Only overlapping pixels are editable; support is `max_j p[c,j] > 0.1`. Ground truth is used only for evaluation after predictions are computed. The runner rejects existing output directories. On the authors' Windows workspace all input datasets and weights stay under E:/Datasets; elsewhere use your existing external dataset root. Do not store datasets or weights in this repository.

The release checks and full-run entry point are kept distinct: packaging reruns algebra/statistical verification, not the 80-image model experiment. See RELEASE_CHECKS.json for the actual checks performed. No local machine paths, manuscript drafts, correspondence or credentials are required in the public repository.

## Repository contents

- sccr.py: frozen unrestricted and support-constrained coordinate solvers.
- metrics.py, io_utils.py, run_voc.py: metrics and portable frozen-protocol runner.
- verify_math.py, reproduce_tables.py: small-problem and table verification.
- results/: per-image metrics, confusion matrices, summary and input hash manifest.
- protocol.json, environment.json, input_checksums.json, confirm_b_ids.txt: reproducibility metadata.
- UPSTREAM.json, NOTICE.md, CITATION.cff: provenance and attribution.

Repository: https://github.com/cdhuangjin/rankseg-sccr. The study's code uses the MIT license selected by the repository owner; see LICENSE and NOTICE.md for attribution and third-party rights.
