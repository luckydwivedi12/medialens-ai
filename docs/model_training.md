# Phase 10: first deepfake image-classification prototype

Run date: 2026-11-15 (Asia/Calcutta). This is a bounded CPU development baseline,
not proof of authenticity, an identity-independent benchmark or a universal
deepfake detector. No locked-test image is opened for training, evaluation,
benchmarking or inference.

## Environment and official dependencies

Confirmed the existing backend virtual environment uses **Python 3.12.10,
64-bit AMD64**. Preflight found approximately 20.1 GB free on C: and 98.1 GB
on D:. The first model output directory did not exist. Existing API/UI
dependencies and previous data phases were preserved.

Installed the exact requested packages from the
[official CPU wheel index](https://download.pytorch.org/whl/cpu):

```powershell
.venv\Scripts\python.exe -m pip install torch==2.14.1 torchvision==0.29.1 --index-url https://download.pytorch.org/whl/cpu
```

Installed versions are **torch 2.14.1+cpu** and **torchvision 0.29.1+cpu**.
`torch.cuda.is_available()` is false and `torch.version.cuda` is null. Both
packages import and a CPU MobileNetV3 forward pass succeeds. No CUDA packages,
TensorFlow or dataset were installed/downloaded. `pip check` passed.

`backend/requirements-ml.txt` pins these exact CPU builds and the official index.
The training report directory contains `environment_freeze.txt` with every
installed package version, plus environment/device/memory metadata. Existing
`requirements.txt` and `requirements-dev.txt` remain unchanged.

## Architecture and pretrained provenance

Use `torchvision.models.mobilenet_v3_large` initialized from
`MobileNet_V3_Large_Weights.IMAGENET1K_V2`. Downloaded only from the specified
[official PyTorch weight URL](https://download.pytorch.org/models/mobilenet_v3_large-5c1a4163.pth).
An initial transient DNS failure was resolved by retrying the same official URL;
no alternate weight source was used.

Pretrained file:
`D:\MediaLensData\models\pretrained\mobilenetv3\mobilenet_v3_large-5c1a4163.pth`.
Full SHA-256:
`5c1a416349c4cf298f2a6a5e2600ed0ee55e604713578f5e74e6bc8bcaef7997`.
The published filename hash prefix and the pinned full file hash are checked.
`model_provenance.json` records source URL, enum, path and checksum.

Replace the final ImageNet classification layer with two logits. Class mapping
is **real=0, fake=1**. Freeze every feature-extractor parameter and keep feature
BatchNorm layers in evaluation mode throughout training. Train the classification
head: its 960→1,280 linear layer, Hardswish, dropout and replacement 1,280→2
linear layer. The entire head is trainable; this is not a last-linear-layer-only
experiment. No backbone fine-tuning is performed.

The two-class model has 4,204,594 parameters; 1,232,642 classifier parameters are
trainable. Feature parameters and buffers are hashed before/after training to
verify that the frozen extractor, including running statistics, stays unchanged.

## Frozen data selection and integrity

Read `docs/final_dataset_split.md`, the actual final Phase 9 report and its
versioned split manifest, and the original Phase 8 crop-pair manifest. Verify
the report/manifests against frozen output checksums and the prospective test
lock. Accept only rows with `paired_subset_candidate=True`,
`selection_status=matched_region_pair_candidate` and `status=crop_saved`.

Every pair must contain exactly one original and one manipulated crop, have the
same frame index/component/partition, agree with the original matched-pair
paths/checksums/video IDs, and retain verified alignment evidence. Reject
cross-split components and duplicate checksums. All 1,472 development images
are path-resolved under external storage, checksum checked and structure-verified.
Test rows are counted and checked as metadata only: their image paths are not
resolved, statted, hashed or opened. The dataset and inference entry points
reject test records before attempting to open an image.

| Paired population | Training | Validation | Locked test, metadata only |
| --- | ---: | ---: | ---: |
| Original/manipulated pairs | 586 | 150 | 118 |
| Images | 1,172 | 300 | 236 |
| Real/fake labels | 586 / 586 | 150 / 150 | 118 / 118 |
| Components represented by these paired images | **65** | **17** | **14** |
| All lineage components in the full Phase 9 cohort | 66 | 17 | 16 |

The image counts match the request exactly. Component counts differ only because
some full-cohort components have no matched pairs: training component 112–892,
and test components 479–706 and 607–683. No extra sample is forced in to make
those groups appear in the paired subset. There are 96 paired-image components
across all metadata partitions, compared with 99 in the full cohort.

Unmatched candidates, bystanders, detector failures, insufficient-support crops
and unresolved records are excluded from this model's dataset. Their original
files and manifests remain intact. Source rows still say
`training_approved=False` and `manual_review_required=True`; Phase 10 explicitly
authorizes prototype training on the matched subset, not a retrospective claim
that those provisional labels received human approval.

## Preprocessing contract

Convert saved crops to RGB, directly resize to **224×224** with antialiased
bilinear interpolation, convert uint8 to float [0,1], and normalize with
mean **[0.485, 0.456, 0.406]** and standard deviation
**[0.229, 0.224, 0.225]** from the pretrained weights.

The [documented ImageNet V2 transform](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.mobilenet_v3_large.html)
normally resizes the shorter side to 232 then center-crops to 224. This prototype
uses the user's explicit direct 224×224 resize instead; that geometry override
and the original documented default are saved in `preprocessing_config.json`.

Training only: horizontal flip probability 0.5; ColorJitter brightness 0.10,
contrast 0.10, saturation 0.05 and hue 0.01. Validation, final training-set
evaluation, smoke inference and latency checks use deterministic preprocessing
with no augmentation. Original image files are never rewritten.

## Bounded training and checkpoint selection

CPU only, float32, batch **8**, DataLoader workers **0**, no pinned-memory/GPU
assumptions. Random/Python/NumPy/PyTorch seed **42** and deterministic PyTorch
algorithms are enabled. AdamW uses learning rate **0.0003** and weight decay
**0.0001**; loss is CrossEntropyLoss.

Benchmark both 4 and 6 intra-op CPU threads on a training-only batch: two warmup
steps and five timed forward/backward/head-update steps per setting. Restore
classifier weights after probing and reseed before real training. Inter-op
threads are 1. The benchmark never trains on validation or test examples.

Measured median step times at batch 8 were **123.66 ms with 4 threads** and
**95.98 ms with 6 threads**, so the run selected **6 threads**. These short,
warm-batch throughput figures exclude image decoding, augmentation and epoch
validation; they are not end-to-end epoch-throughput claims.

At most **5 epochs**. Select the checkpoint with lowest validation loss.
Early stopping patience is **2** epochs without a meaningful loss improvement
of at least **0.0001**; even smaller new minima are still checkpointed as the
lowest loss. Validation accuracy and AUC are reported, not used to override
the loss-based choice. The binary decision threshold is fixed at **0.5**.
No checkpoint, augmentation or threshold is selected against test data.

Write one history row and validation-metrics JSON per completed epoch.
`augmented_training_loss` in history is an online loss measured while training
weights change; final training metrics are a separate, unaugmented full pass of
the selected checkpoint and should not be confused with that online statistic.

Improving epoch checkpoints are uniquely named. Checkpoint serialization uses
an exclusive temporary file, flush/fsync and atomic no-overwrite hard-link
publication. The final `best_model.pth` is published the same way. Existing model
directories are preserved; subsequent runs create unique subdirectories under
`models\medialens_mobilenetv3\runs` instead of replacing a checkpoint.
Checkpoint loading uses CPU mapping, `weights_only=True`, strict state loading,
and checks of the architecture, class mapping and preprocessing contract.

Safety bounds are a 30-minute training budget, 4 GiB process-RSS budget and
1 GiB minimum available system RAM. Check these at every training batch; preserve
the last validated checkpoint if a bounded run stops. Peak/current working set
and available system memory are read through Windows APIs without extra packages.
No temperature sensor is available through this runtime, and no CPU-temperature
measurement or thermal guarantee is claimed.

## Metrics, inference and interpretation

Evaluate the saved best checkpoint on all training and validation images only.
Report cross-entropy loss, accuracy, precision, recall, F1, specificity, ROC-AUC
and confusion matrices. Fake is positive. Confusion matrices use true-label
rows and predicted-label columns, both ordered [real, fake]. AUC uses ranked
positive-class softmax scores with average ranks for ties; it does not require
scikit-learn. Undefined AUC for a one-class population is null, not invented.

Reload the saved checkpoint and run four smoke samples: one pair from training
and one from validation. Verify two output logits, the saved class mapping,
finite softmax values summing to one, shared preprocessing and exactly repeated
CPU logits. Measure batch-1 network and decode/preprocess/network latency
separately, after warmup, on a validation image. These model scores are not
calibrated probabilities of real-world authenticity.

## Reproduction

From `backend`, install base requirements and then the optional CPU model stack:

```powershell
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe -m pip install -r requirements-ml.txt
.venv\Scripts\python.exe -m pip check
.venv\Scripts\python.exe scripts\train_baseline.py `
  --split-report D:\MediaLensData\reports\dataset_split_report_phase9_20261115T104516317959Z_8f2f4d7e21b84ff795f566fef7452c03.json `
  --storage-root D:\MediaLensData `
  --epochs 5 --batch-size 8 --threads 6 --max-training-seconds 1800
```

The `--threads 6` option reproduces the recorded setting while still benchmarking
both thread counts. Omit it for a new measured thread choice. Reproduction uses
the same seed, data hashes and configuration; timing and floating-point behavior
across different hardware/library/thread configurations are not guaranteed
bit-identical. The report's complete environment freeze records transitive
versions as well as the top-level CPU pins.

For a separate saved-model timing check:

```powershell
.venv\Scripts\python.exe scripts\benchmark_model.py `
  --checkpoint D:\MediaLensData\models\medialens_mobilenetv3\best_model.pth `
  --split-report D:\MediaLensData\reports\dataset_split_report_phase9_20261115T104516317959Z_8f2f4d7e21b84ff795f566fef7452c03.json `
  --output-directory D:\MediaLensData\reports\training `
  --threads 6
```

Each invocation writes a uniquely named timing report. The benchmark opens only
the development data accepted by the shared loader. Backend tests use local
temporary fixtures and untrained model construction; they do not download
weights or access dataset test pixels. Run `.venv\Scripts\python.exe -m pytest`
from `backend` and `npm run build` from `frontend`.

## Known limits

The paired region labels are still provisional video-lineage annotations, not
human-certified face truth. Crops/frames within a component are correlated;
crop-level metrics do not count independent people. Person-identity independence
is unverified. The cohort has one source dataset, Deepfakes method and c23
compression, so results do not establish cross-dataset, unseen-method, audio or
fully generated-image performance. Broadcast overlays and acquisition artifacts
may support shortcuts. The ImageNet feature extractor can retain pretraining
biases; no audit proves independence of its upstream pretraining data.

The Phase 9 test lock is prospective: Phase 8 computed whole-cohort preparation
statistics before this split was frozen. Phase 10 performs no locked-test
evaluation or tuning. A later authorized final evaluation needs its own protocol;
a pristine benchmark would require separately reserved data and a frozen
preparation process. Human label review, probability calibration and broader
evaluation remain future work. No frontend or API inference integration is added.

## Actual completed run

Run ID: `phase10_20261115T111020125779Z_6b17e16655ea4b4aa70496c601f90d28`.
Completed **5 epochs** in **196.45 seconds** (3 minutes 16.45 seconds).
The complete scripted run took **228.67 seconds**, including development-image
verification, thread benchmarking, checkpoint reload, final evaluation and smoke
checks. These timings exclude the earlier package installation and initial
pretrained-weight download.

The lowest validation loss occurred at **epoch 3**. Epochs 4 and 5 did not improve
it, meeting early-stopping patience at the same time as the five-epoch maximum.
The saved best model is epoch 3 even though epoch 4 has slightly higher accuracy
and epoch 5 has higher AUC; checkpoint selection follows the declared loss rule.

| Epoch | Online augmented train loss | Validation loss | Validation accuracy | Validation AUC | Seconds |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.706791 | 0.631336 | 67.33% | 0.740756 | 41.34 |
| 2 | 0.625965 | 0.645220 | 63.00% | 0.767289 | 40.51 |
| **3, selected** | **0.584742** | **0.590172** | **69.67%** | **0.778933** | **37.31** |
| 4 | 0.529195 | 0.620982 | 70.00% | 0.777156 | 39.78 |
| 5 | 0.499362 | 0.608371 | 68.33% | 0.802089 | 37.50 |

Full, deterministic evaluation of the reloaded **epoch-3 checkpoint**:

| Metric | Training, 1,172 images | Validation, 300 images |
| --- | ---: | ---: |
| Cross-entropy loss | 0.504338 | 0.590172 |
| Accuracy | 77.30% | 69.67% |
| Precision, fake positive | 80.53% | 78.64% |
| Recall, fake positive | 72.01% | **54.00%** |
| F1 | 0.760360 | 0.640316 |
| Specificity | 82.59% | 85.33% |
| ROC-AUC | 0.872354 | 0.778933 |

Training confusion matrix, true rows/predicted columns [real, fake]:
`[[484, 102], [164, 422]]`.
Validation confusion matrix: `[[128, 22], [69, 81]]`.
The validation model misses 69 of 150 manipulated-region candidates at the fixed
0.5 threshold. This is a modest development baseline, not production readiness.
No threshold was tuned to improve that result.

Peak process working set during the training epochs was **486,400,000 bytes
(463.87 MiB)**. Peak for the complete process, including checkpoint reload and
final evaluation, was **542,572,544 bytes (517.44 MiB)**. The recorded available
physical RAM snapshots stayed above 4.1 GB. No RAM or runtime guard fired; batch
size remained 8. No temperature data was available.

At 6 threads and batch 1, after three warmup forwards, 15-repeat latency checks
measured **22.56 ms median / 24.09 ms p95** for the network alone and **21.98 ms
median / 32.28 ms p95** for PNG decode, preprocessing, network and softmax. The
two timing series were measured separately; scheduling/cache noise explains
the small median inversion. Do not subtract those medians to infer negative
preprocessing time, or equate classifier-only timing to full video-pipeline speed.

Saved-checkpoint smoke checks passed for two training and two validation crops:
CPU shape [1,2], real/fake mapping, finite probabilities summing to one, frozen
preprocessing and exact repeated logits. Feature parameter/buffer digests agree
before training and after checkpoint reload. **Zero locked-test images were
opened or evaluated.**

## Saved outputs and required checks

Model directory: `D:\MediaLensData\models\medialens_mobilenetv3`.

- `best_model.pth` — selected epoch 3, 26,892,021 bytes, including model,
  optimizer/RNG state and primitive metadata.
- `model_metadata.json` — architecture, weights, data provenance, classes,
  parameter counts, selected epoch, checkpoint checksum and limitations.
- `preprocessing_config.json` — exact RGB/geometry/normalization/augmentation contract.
- `epoch_001.pth` and `epoch_003.pth` — preserved improving checkpoints.

Best checkpoint SHA-256:
`5e19cc61d7b0c74b1f7979fb0682bc510dbc781c97d06f77945c50a25ab58622`.

Report directory:
`D:\MediaLensData\reports\training\phase10_20261115T111020125779Z_6b17e16655ea4b4aa70496c601f90d28`.
It contains `training_history.csv`, `training_metrics.json`,
`validation_metrics.json`, per-epoch validation JSONs,
`training_predictions.csv`, `validation_predictions.csv`,
`training_summary.json`, `model_provenance.json`, `dataset_provenance.json`,
`environment.json`, `environment_freeze.txt`, `cpu_thread_benchmark.json`,
`inference_smoke_test.json`, `inference_latency.json` and
`artifact_checksums.json`. All are external; no model or data artifact is
stored in Git. Existing checkpoints and previous-phase outputs were not overwritten.

Final verification: **117 backend tests passed** (including 11 new classifier
tests), with the existing Starlette/httpx deprecation warning; **frontend
production build passed**; **pip check passed**. Tests cover dataset RGB loading,
normalization and class mapping, deterministic validation preprocessing,
augmentation/test access guards, binary output shape, frozen feature state,
atomic checkpoint roundtrip/no-overwrite behavior, metadata tampering, metric
definitions/ties and bounded training configuration.

Future work is human review of candidate labels and failure modes, followed by
separately authorized calibration/broader evaluation. Any backbone fine-tuning,
locked-test evaluation or frontend/API integration requires a later phase.
Phase 10 is complete; no later phase has started.
