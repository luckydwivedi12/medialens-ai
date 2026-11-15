# Phase 6: training data readiness and model design

Assessment date: **2026-11-15, Asia/Calcutta**. All quantities below are marked by
context as measured, official reference values, or proposed estimates. No accuracy
has been measured. This phase adds documentation only; it does not acquire data,
weights or packages, train a model, or assign any split.

**Readiness decision: pipeline smoke-test dataset.** The saved images are readable
and traceable, but ten filename-derived groups and inherited video labels are
insufficient for credible deepfake classification accuracy. Resolve target-face
label validity and expand source diversity before a research baseline. A later
optimization smoke test must not be reported as a benchmark.

## 1. Evidence inspected and measurement method

Read AGENTS.md, dataset_strategy.md, faceforensics_sample.md and
frame_extraction_pipeline.md. Inspected these actual external artifacts:

- Video manifest: `D:\MediaLensData\manifests\faceforensics_sample_manifest.csv`.
- Video inventory: `D:\MediaLensData\reports\dataset_inventory_20261115T084002294908Z_d2d3bf0c8fbc4866b8b88e31124f4e51.csv`.
- Frame manifest: `D:\MediaLensData\manifests\frame_extraction_20261115T085602795863Z_ab2d90f7dbc34410880fa34482f99d1d.csv`.
- Face manifest: `D:\MediaLensData\manifests\face_detection_20261115T085602795863Z_ab2d90f7dbc34410880fa34482f99d1d.csv`.
- Summary: `D:\MediaLensData\manifests\processing_summary_20261115T085602795863Z_ab2d90f7dbc34410880fa34482f99d1d.json`.
- Report: `D:\MediaLensData\reports\processing_report_20261115T085602795863Z_ab2d90f7dbc34410880fa34482f99d1d.json`.
- Actual images in `D:\MediaLensData\processed\face_crops\20261115T085602795863Z_ab2d90f7dbc34410880fa34482f99d1d`.

Class counts were obtained from individual manifest rows and matched to their
source videos, not inferred by dividing 204 in half. All 204 crop files were
structure-verified and decoded, checked for square dimensions and boundary-valid
crop coordinates, and hashed against their records. Decoded RGB pixels plus image
dimensions were independently hashed to check exact pixel duplicates. All 200
parent-frame hashes and all 40 raw-video hashes were checked; video inventory
hashes agree with the source manifest. The summary and report contain identical
JSON data.

Quality measurements use native-resolution OpenCV grayscale images: mean intensity
for brightness, pixel standard deviation for contrast, and variance of the default
3x3-neighborhood discrete Laplacian operation (`cv2.Laplacian(gray, CV_64F)`, default
ksize=1) as a sharpness proxy. The latter is not a validated blur classifier and
depends on resolution, texture, noise and compression. Percentiles use NumPy's
default linear interpolation. No image was changed or automatically excluded.

## 2. Actual distribution and provenance

| Measured item | Result |
| --- | ---: |
| Saved crop files / successful face-manifest rows | 204 / 204 |
| Inherited real-video crop labels | **102 (50%)** |
| Inherited fake-video crop labels | **102 (50%)** |
| Source videos represented | 40: 20 real, 20 fake |
| Extracted parent frames | 200: 100 real-video, 100 fake-video |
| Filename-derived leakage groups | **10** |
| Crops with unassigned split | 204 |
| Missing or unexpected crop files | 0 / 0 |
| Corrupt/unreadable crop images | 0 |
| Asset or parent hash discrepancies | 0 |
| Exact duplicate byte-hash groups | 0 |
| Exact duplicate decoded-pixel groups | 0 |
| Missing required metadata or inherited-field mismatches | 0 |

Each group represents two originals and both reciprocal fakes. These are ten
disjoint components under known sequence relationships, **not ten proven
independent identities or upstream recordings**. Identity and upstream YouTube
overlap have not been established. More crops from existing videos do not add
independent evidence.

The table gives all 40 videos and their actual crop counts. Prefix each pair with
`ffpp-c23-pair-` to obtain its full leakage_group_id.

| Pair/group suffix | Original videos (crop count) | Fake videos (crop count) | Group crops |
| --- | --- | --- | ---: |
| 033-097 | 033 (5), 097 (5) | 033_097 (6), 097_033 (5) | 21 |
| 183-253 | 183 (5), 253 (5) | 183_253 (5), 253_183 (5) | 20 |
| 210-241 | 210 (5), 241 (5) | 210_241 (5), 241_210 (5) | 20 |
| 252-266 | 252 (5), 266 (5) | 252_266 (5), 266_252 (5) | 20 |
| 339-392 | 339 (5), 392 (5) | 339_392 (5), 392_339 (5) | 20 |
| 469-481 | 469 (5), 481 (5) | 469_481 (5), 481_469 (5) | 20 |
| 585-599 | 585 (7), 599 (5) | 585_599 (6), 599_585 (5) | 23 |
| 672-720 | 672 (5), 720 (5) | 672_720 (5), 720_672 (5) | 20 |
| 828-830 | 828 (5), 830 (5) | 828_830 (5), 830_828 (5) | 20 |
| 866-878 | 866 (5), 878 (5) | 866_878 (5), 878_866 (5) | 20 |

### Image quality distribution

All 204 assets are RGB PNGs with square native dimensions. Side length ranges
from **50 to 562 pixels**; median 323; 25th/75th percentiles 190/365.75; mean 310.36.

| Native side length | Real-video labels | Fake-video labels | Total |
| --- | ---: | ---: | ---: |
| Below 64 | 0 | 1 | 1 |
| 64-127 | 6 | 7 | 13 |
| 128-223 | 26 | 24 | 50 |
| At least 224 | 70 | 70 | 140 |

Thus **64/204 crops (31.37%)** need upsampling to reach a 224-pixel model input;
resizing cannot recover missing facial detail. The 50x50 crop is
`033_097_frame_00000242_face_001.png`, confidence 0.900205. It requires target/quality
review, not an automatic relabel or deletion. Fifteen crop rectangles touch at
least one original-frame boundary. Boundary contact alone does not prove that the
face is clipped or unusable.

| Native-image diagnostic | Minimum | P25 | Median | P75 | Maximum |
| --- | ---: | ---: | ---: | ---: | ---: |
| Grayscale mean, 0-255 | 27.66 | 74.05 | 94.18 | 112.41 | 181.46 |
| Grayscale standard deviation | 25.17 | 41.03 | 47.72 | 56.97 | 81.29 |
| Laplacian variance, proxy only | 7.47 | 36.34 | 103.78 | 263.93 | 1,834.52 |

Real/fake median Laplacian variance is 102.77/103.78 and mean brightness is
96.23/96.63. These small-sample similarities do not establish quality equivalence
or absence of shortcuts. No universal blur threshold or manual face-quality
certification is claimed. Future quality decisions need reviewed evidence and
must not be chosen from final-test classifier performance.

### Face detection and label quality

YuNet found 204 faces in 200 sampled frames, with zero recorded no-face events,
frame failures or processing errors. Four frames have two detections:
`033_097:242`, `585_599:179`, `585:179`, `585:323`. The rest have one. This gives
1.02 faces/frame; it does not prove that all faces were detected correctly.

| Localization confidence statistic | All | Real-video crops | Fake-video crops |
| --- | ---: | ---: | ---: |
| Minimum | 0.900050 | 0.900050 | 0.900205 |
| Mean | 0.940494 | 0.940424 | 0.940564 |
| Median | 0.942195 | 0.942153 | 0.942195 |
| Maximum | 0.956384 | 0.955583 | 0.956384 |

Confidence bins: [0.90,0.92): 6; [0.92,0.94): 79; [0.94,0.96): 119;
[0.96,1.00]: 0. These are detector localization scores, not authenticity or
manipulation probabilities.

The 42-column derived manifests retain all original provenance fields plus asset,
timestamp, detector and crop metadata. Required values and relationships are
complete; real-video donor IDs are correctly blank. All fake-video donor IDs
resolve to the expected original/group. Checksums identify video, frame and crop
assets separately. Five landmark pairs and valid confidence values were present
on every saved face row.

**Label-validity blocker:** `label=fake` currently means the parent video is fake.
It does not prove the particular crop depicts the manipulated target. All rows
explicitly say `inherited_video_label_not_face_ground_truth`. Review the four
multi-face frames, the tiny crop and a stratified sample of single-face frames;
ultimately validate the training target for every selected fake crop through
reviewed target tracks or authorized manipulation annotations. Treat ambiguous
crops as ineligible until resolved in a separate selection manifest. Preserve
original files and annotations. A high-confidence detector false positive can
also be mislabeled; confidence is not a substitute for target review.

Exact duplicate checks passed, but no near-duplicate clustering, identity audit
or cross-dataset source audit was performed in Phase 6. Adjacent frames and
real/fake counterparts can share almost all content without equal hashes.

## 3. Resource-aware acquisition plan (not executed)

Official dataset documentation was rechecked on 2026-11-15:
[FaceForensics dataset README](https://github.com/ondyari/FaceForensics/blob/master/dataset/README.md).
The local author-supplied script was read as text, never executed:
`D:\MediaLensData\tools\faceforensics_download_v4.py`.
Its SHA-256 is
`3d0aa3a186ec2f210056cdfcc5fba97d6f77f7872cfafd1754063954683a47ba`.
The workspace copy differs only in the final newline after text normalization;
neither copy was changed. Its recorded official origin is the
[author download-script URL](http://kaldir.vc.in.tum.de/faceforensics_download_v4.py)
provided in the owner's access email. No remote script/data file was saved here.

Verified script behavior:

- Defaults are dataset=all and compression=raw; both are unsuitable for a bounded
  c23 Deepfakes-only expansion. Specify every option explicitly.
- `-d original` flattens each server file-list pair into its two original IDs.
  `-d Deepfakes -t videos` emits both A_B and B_A for each pair, in pair order.
- `-n` / `--num_videos` truncates the generated list to a count of **videos**, not
  groups. Use the same positive even count for originals and Deepfakes. Odd counts
  can leave an incomplete reciprocal group. Zero/negative values do not disable
  acquisition: the script applies a limit only when the value is positive.
- The script has no explicit ID-list or official-split selection argument.
  It uses the server list's first N entries; it does not make a random selection
  or ensure representative train/validation/test coverage.
- EU2 uses the documented author-hosted endpoint. Existing files are skipped by
  filename, not reverified by checksum. Independently audit all retained files.
- It imports tqdm. Reuse the owner's existing downloader environment in a later
  acquisition phase; do not silently add this dependency to the backend now.

**First expansion recommendation: 100 total reciprocal pair groups, nominally
200 originals + 200 Deepfakes videos (400 total), including the existing sample
if its IDs are in the frozen server list.** This is ten times the current group
count, not 400 independent videos. It adds nominally 90 groups/360 videos, subject
to lineage and duplicate checks. It is a development cohort, not a promised final
benchmark. Acquire more groups before increasing frames per existing video.

For a later authorized acquisition, the bounded commands would be:

```powershell
python "D:\MediaLensData\tools\faceforensics_download_v4.py" "D:\MediaLensData\raw\faceforensics" -d original -c c23 -t videos --num_videos 200 --server EU2
python "D:\MediaLensData\tools\faceforensics_download_v4.py" "D:\MediaLensData\raw\faceforensics" -d Deepfakes -c c23 -t videos --num_videos 200 --server EU2
```

**These commands were not run.** Before using them, freeze and inspect the actual
server pair list and official split IDs under authorized access. Confirm complete,
unique pairs, original closure, current-sample overlap and split coverage. Server
ordering is not guaranteed stable between calls. If that prefix cannot meet the
experiment's split/diversity requirements, stop and design a separately reviewed
explicit-ID acquisition procedure; do not pretend the script supports a filter
it does not have. Preserve the source script and old manifests/reports.

### Storage grounded in measured files

D: snapshot: **100,521,627,648 free bytes = 93.62 GiB**. Free space is not reserved.
Current measured storage:

| Asset | Count | Bytes | Mean bytes/asset |
| --- | ---: | ---: | ---: |
| Original c23 videos | 20 | 38,690,892 | 1,934,544.60 |
| Deepfakes c23 videos | 20 | 39,511,033 | 1,975,551.65 |
| Extracted full-frame PNGs | 200 | 136,583,110 | 682,915.55 |
| Face-crop PNGs | 204 | 28,580,133 | 140,098.69 |

Raw c23 media total: 78,201,925 bytes. Current processed run including manifests
and report copies: 165,828,737 bytes. Per observed four-video group: 7,820,192.5
raw bytes + 16,582,873.7 processed bytes. This sample spans raw video sizes from
618,639 to 7,562,076 bytes, so the mean is not a per-file size guarantee.

Using those actual means and the same five-frame pipeline, estimates are:

| Total nominal groups | Original + fake videos | Raw c23 GB | Processed GB | Combined GB |
| --- | --- | ---: | ---: | ---: |
| 100 (first milestone) | 200 + 200 | 0.782 | 1.658 | 2.440 |
| 250 (later development) | 500 + 500 | 1.955 | 4.146 | 6.101 |
| 500 (core paired inventory, if validated) | 1,000 + 1,000 | 3.910 | 8.291 | 12.202 |

GB here is decimal. The first milestone implies roughly 2,000 frames and 2,040
crops **only if** current yield holds; failures/extra faces can change that count.
Incremental storage above the current retained sample/run is about **2.196 GB**.
Reserve **8 GB of growth budget** as a rounded three-times-total buffer, and keep
at least **20 GiB free**. This covers variation and some staging/reruns, not unlimited
duplicate runs. Recheck actual size after each acquired batch. Do not delete old
data to meet estimates without a separate decision. A later full paired inventory
has a three-times-estimate budget of ~36.6 GB, but is not authorized here and CPU
processing remains a separate constraint.

Compression policy: keep **c23** as the first experiment's acquisition and
processing variant; original and manipulated controls must use the same setting.
`raw`/c0 is lossless video, not an extracted-image directory. The official README
estimates the much broader full FF++ raw video release at ~500 GB and all PNG
frames at ~2 TB; neither fits this free-space budget. No raw/c0 subset was measured,
so do not extrapolate c0 byte requirements from c23. Defer it to a separately
budgeted, complete-group robustness experiment. Treat compression variants as
the same lineage group, never independent split samples. Do not use `-d all`,
download additional manipulation techniques, Deepfakes generator weights, or bulk
frames for this first expansion. The intended classifier's scope remains Deepfakes
face swapping at c23, not every deepfake technique.

## 4. Candidate architecture comparison

Official TorchVision model pages checked on 2026-11-15 (displayed version 0.29):
[EfficientNet-B0](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.efficientnet_b0.html),
[ResNet50](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.resnet50.html),
[MobileNetV3-Large](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.mobilenet_v3_large.html).
No builders were executed and no weights were retrieved. Choose a compatible
PyTorch/TorchVision CPU pair only in a later installation phase.

| Model | Official 1,000-class parameters | Documented GFLOPs | FP32 parameter storage, derived | Available pretrained weights / planned input |
| --- | ---: | ---: | ---: | --- |
| EfficientNet-B0 | 5,288,548 | 0.39 | 20.17 MiB | ImageNet-1K V1; RGB 224x224 |
| ResNet50 | 25,557,032 | 4.09 | 97.49 MiB | ImageNet-1K V1/V2; choose explicit V2; RGB 224x224 |
| MobileNetV3-Large | 5,483,032 | 0.22 | 20.92 MiB | ImageNet-1K V1/V2; choose explicit V2; RGB 224x224 |

The official GFLOPs describe reference image-classification models, not measured
latency on this laptop or the future binary heads. Depthwise convolutions,
threading and kernel implementations mean fewer operations do not guarantee a
proportional latency reduction. File-size metadata is ~20.5/97.8/21.1 MB,
respectively, and differs from raw tensor arithmetic because of serialization
and buffers. ImageNet classification results are not evidence of deepfake accuracy.

- **EfficientNet-B0:** compact, moderate reference compute and a reasonable second
  baseline. Its depthwise/MBConv blocks and activation memory still need CPU
  benchmarking; image-quality and compression shortcuts remain possible.
- **ResNet50:** established transfer-learning control with larger representational
  capacity, but substantially greater parameter/compute cost. Defer full fine-tuning
  on this CPU; capacity cannot compensate for ten groups and can memorize sources.
- **MobileNetV3-Large:** lowest reference compute here with similar original model
  storage to B0, making frozen-feature transfer learning the best first hardware
  fit to test. Mobile-oriented operations are not guaranteed fastest on this exact
  CPU; limited fine-detail capacity and resize sensitivity are research risks.

**Recommendation: MobileNetV3-Large, explicit IMAGENET1K_V2 transfer learning.**
Replace only the final 1280-to-1000 layer with a 1280-to-2 linear layer, preserving
the pretrained intermediate classifier. Freeze pretrained parameters first.
This yields **4,204,594 total parameters by arithmetic**, not by instantiation;
only **2,562** new output-layer parameters are trainable initially. Later consider
unfreezing the last feature stage only after a measured CPU/validation trial.
Keep BatchNorm running statistics frozen with the feature extractor in eval mode;
disable autograd through frozen features. Do not accidentally call train() on all
frozen BatchNorm layers. Pretrained transfer learning is preferable to scratch
training on such limited source diversity. [Official transfer-learning tutorial](https://docs.pytorch.org/tutorials/beginner/transfer_learning_tutorial.html)

All three models can learn identity, background, detector, resize or codec cues
instead of manipulation. ImageNet pretraining is not face-forgery specialization.
Domain shift and unseen synthesis remain problems regardless of model choice.
Published model weights also carry their own use considerations; before future
acquisition, record the exact enum, source, hash and applicable terms rather than
assuming the code license settles pretrained-data rights.

## 5. Proposed training methodology — design only

### Data eligibility and grouped splitting

1. Resolve target-face ambiguity and freeze a new eligibility/selection manifest
   referencing unchanged originals. Record exclusions and reviewed reasons. Match
   original/fake crop policies; do not filter one class more aggressively to make
   a cleaner-looking training set. Count videos, groups and crop coverage after
   selection, including no-face and ambiguity failures.
2. Build connected components joining target and donor originals, reciprocal fakes,
   all frames/crops, recompressions, confirmed duplicates and common upstream
   recordings. Add reliable identity relationships when available. Audit global
   near-duplicate/source overlap before freezing components. Never split crops
   randomly. Every derived asset inherits its parent component's assignment.
3. Prefer the official FF++ train/validation/test ID lists, described in the
   [original paper](https://arxiv.org/html/1901.08971v3) as 720/140/140 original
   sequences. Audit that all component members agree. Quarantine a component that
   crosses official boundaries; report the reduced cohort and departure from
   exact benchmark reproduction. Do not move test members into training.
4. If suitable official splits cannot be used, predeclare a separate deterministic
   component-level 70/15/15 protocol with seed 42 and group-level class/method
   balance where feasible. The ratios are proposed, not assignments. Hash the
   manifests, membership, duplicate audit and configuration before model work.
5. Keep the present ten-group cohort entirely unassigned. At 100 clean groups,
   a hypothetical 70/15/15 division would still leave only 15 validation and 15
   test groups, giving unstable estimates. Expand toward hundreds of audited
   groups (potentially the full paired original/Deepfakes collection), with
   substantial clean held-out coverage, then an independently collected dataset.
   There is no universal group count that makes accuracy credible automatically.

Only ten groups cannot establish stable source-generalizing classification.
Additional faces/frames from them increase correlated observations, not independent
test cases. At expanded scale, verify diversity in identities, recording conditions,
face size and compression; confirm valid official split coverage rather than
assuming a first-N acquisition produces it. Report group-level uncertainty.

### Inputs, class balance and augmentation

- Decode RGB explicitly: OpenCV images are BGR and must be converted. Keep saved
  crops unchanged; transform in the future loader.
- For MobileNetV3-Large V2 validation/test, follow its documented preset: bilinear
  resize of the square crop to 232, center crop to 224, float range [0,1], then
  mean [0.485,0.456,0.406] and std [0.229,0.224,0.225]. Preserve this exact recorded
  policy across inference and evaluation. B0's official V1 preset instead uses
  bicubic resize 256; ResNet50 V2 uses bilinear resize 232. Do not mix presets
  silently. Upsampling low-resolution crops does not create information.
- Proposed train-only augmentation: horizontal flip p=0.5, mild scale/translation
  retaining most facial context (e.g. crop scale 0.9-1.0), and brightness/contrast
  jitter up to 0.1. Start with a small, fixed recipe; no random validation/test
  transforms. Avoid heavy blur, smoothing or compression that may erase forensic
  cues. Any later JPEG/noise robustness augmentation must be class-symmetric,
  versioned and evaluated through validation, never tuned on final tests.
- Present 102/102 balance requires no class reweighting. After expansion/eligibility
  selection, count imbalance using **training only**. Start with a cap/equalization
  on crops per source video so multi-face videos do not dominate. If needed, use
  weighted cross-entropy with class weights N/(2*N_class), computed from eligible
  training records. Do not blindly combine inverse-frequency sampling and weighted
  loss, which can compensate twice. Preserve natural held-out distributions.

### Optimization, stopping and thresholds

Proposed initial configuration, to be benchmarked and validated later:

- Map real=0, fake=1. Produce two logits. Use cross-entropy on logits and integer
  class targets; do not apply softmax before the loss. Use softmax's fake-class
  score only for scoring/thresholding. [CrossEntropyLoss documentation](https://docs.pytorch.org/docs/2.15/generated/torch.nn.CrossEntropyLoss.html)
- AdamW for trainable parameters, initial head LR 1e-3, weight decay 1e-4,
  betas (0.9,0.999). A later last-stage fine-tuning trial can use 1e-5 for that
  feature stage and 1e-4 for the head; treat this as a new recorded stage, retain
  the best validated checkpoint and do not unfreeze the full backbone by default.
  These are starting proposals, not tuned values. [AdamW](https://docs.pytorch.org/docs/2.15/generated/torch.optim.AdamW.html)
- ReduceLROnPlateau on validation cross-entropy, mode=min, factor=0.5, patience=2,
  min_lr=1e-6. Set an initial ceiling of 20 epochs/stage and early-stop after five
  epochs without a meaningful validation-loss improvement (proposed absolute
  delta=1e-4). Persist optimizer/scheduler/RNG state and select the checkpoint
  with minimum validation loss. Never select using test performance.
  [Scheduler documentation](https://docs.pytorch.org/docs/2.15/generated/torch.optim.lr_scheduler.ReduceLROnPlateau.html)
- Select the decision threshold on validation data only, after checkpoint choice.
  Predeclare a criterion such as maximum balanced accuracy, with a deterministic
  tie rule; report the 0.5 reference operating point too. If operational false
  positives matter, define a validation-based specificity target instead before
  tuning. Do not choose the criterion after inspecting final-test results.
- Freeze model, input transforms, aggregation, eligibility rules and threshold
  before the locked test. A poor test result is a result, not permission to retune.
  If temperature calibration is later needed, reserve validation/calibration data
  within development; never fit calibration on the final test.

### Evaluation and cross-dataset design

Use fake as the positive class. Report ROC-AUC from continuous scores, precision
TP/(TP+FP), recall TP/(TP+FN), F1, specificity TN/(TN+FP), and a labeled confusion
matrix with fixed axes true [real,fake] x predicted [real,fake]. Include support,
threshold and class prevalence. Undefined metrics must be reported as undefined,
not silently replaced with an impressive value. Also report PR-AUC and balanced
accuracy when imbalance makes raw accuracy misleading.

Report crop-level metrics only on verified target-eligible face crops, and a
predeclared video-level result using a fixed frame/target-track aggregation policy
(proposed mean fake probability across eligible sampled target frames). Do not
arbitrarily average all bystanders. If target identity or sufficient face coverage
cannot be established, abstain and report coverage/failed-video counts. Bootstrap
confidence intervals by the full lineage group, not by individual crop. Inference
preprocessing/detection time belongs in end-to-end latency, not classifier time.

Reserve Celeb-DF v2 for independent evaluation after separate access/terms and
derived-data permission are confirmed. Exclude it from training, normalization
fitting, model/augmentation selection, calibration and threshold tuning. Audit
source/identity/duplicate overlap with FF++ and use the official evaluation list
once verified. Retain the FF++ validation-selected threshold; report domain shift
and coverage. FF++ trained only on Deepfakes/c23 does not justify claims about
Face2Face, FaceSwap, NeuralTextures, audio or fully generated scenes. A later
unseen-method study needs its own authorized, held-out cohort. DeeperForensics
is not automatically independent because its target videos overlap FF++.

## 6. Compute and memory estimates

No PyTorch is installed, and no training or inference benchmark was run in Phase 6.
The 36.2-second Phase 5 preprocessing run measures YuNet/PNG work, not classifier
training or inference. Do not extrapolate a training duration from it.

FP32 parameter calculations after replacing just the final layer with two outputs:

| Model | Derived binary parameters | Weights only | Weights + gradients + two Adam moment arrays |
| --- | ---: | ---: | ---: |
| EfficientNet-B0 | 4,010,110 | 15.30 MiB | 61.19 MiB |
| ResNet50 | 23,512,130 | 89.69 MiB | 358.77 MiB |
| MobileNetV3-Large | 4,204,594 | 16.04 MiB | 64.16 MiB |

Formula: parameter_count*4 bytes for weights, *16 for the four FP32 arrays when
all parameters are trainable with ordinary AdamW. Counts are derived from official
classifier structures, not measured allocations:
[EfficientNet source](https://github.com/pytorch/vision/blob/main/torchvision/models/efficientnet.py),
[ResNet source](https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py),
[MobileNetV3 source](https://github.com/pytorch/vision/blob/main/torchvision/models/mobilenetv3.py).
The first MobileNet stage trains only 2,562 parameters, so backbone gradient/moment
arrays are unnecessary. Activations, intermediate classifier tensors, BatchNorm
buffers, Python objects, decoder buffers, library runtime, allocator caches and
optimizer workspaces are **additional**. Parameter bytes are not total RAM use.

One float32 RGB 224x224 tensor is 602,112 bytes (~0.574 MiB). Input-only batches
of 4/8/16 take ~2.30/4.59/9.19 MiB; training activations can exceed this substantially.
Start with **batch 8 for a frozen-feature head**, reduce to 4 for a last-stage
fine-tuning trial, and benchmark inference at batch 1 plus small throughput batches.
Try 16 only after measuring resident/peak memory and throughput. Begin with
num_workers=0 on Windows, no pinned-memory assumption and no whole-dataset cache;
test one worker only if decoding becomes a measured bottleneck. Benchmark 2-4 CPU
threads rather than assuming all logical threads help the hybrid i5 CPU.

Planning allowances, not measured RSS: reserve roughly 0.5-1.5 GiB for Python and
loaded vision/ML runtime before significant training buffers; start with a
**2-4 GiB training-process budget**, measure actual peak RSS during the first
authorized batches, and aim below 6 GiB so Windows, the desktop and shared iGPU
memory retain headroom. These allowances are not hard predictions; stop/reduce
batch/workers if paging or unexpected allocation appears. Reserve at least ~6 GiB
for OS/other apps and avoid simultaneous large preprocessing/training jobs.

There is no NVIDIA CUDA GPU. The Iris Xe is not assumed to accelerate a standard
CPU PyTorch setup; no alternative acceleration stack is installed or proposed for
this phase. CPU backpropagation, sustained power/thermal limits, decode costs and
thread contention constrain experiments. Start with frozen transfer learning,
record warmup-excluded latency percentiles, images/second, epoch work, RSS and
thread counts before promising training time or real-time video throughput.
Store future approved weights, checkpoints and caches under D:\MediaLensData,
outside Git. Do not install or download them as part of this plan.

## 7. Next implementation steps and stop boundary

1. Review target-face eligibility and record ambiguity without rewriting original
   labels or images; define comparable original/fake crop selection.
2. In a separately authorized acquisition phase, freeze official pair/split
   metadata and acquire the bounded original/Deepfakes c23 expansion. Reaudit
   complete groups, hashes, labels, duplicates, quality, size and split coverage.
3. Design and freeze audited grouped split/eligibility manifests for the expanded
   cohort; the current files remain unassigned. Budget a broader held-out and
   independent dataset before benchmark claims.
4. Only after an explicit implementation request, choose compatible CPU PyTorch/
   TorchVision versions, acquire the specified pretrained weights with provenance,
   and implement a tested manifest-driven RGB loader and MobileNet baseline.
5. Benchmark memory/throughput before a bounded head-training experiment. Select
   checkpoints/thresholds through validation only and preserve the locked test.

No current model accuracy, authenticity guarantee, split assignment, PyTorch
installation, pretrained-weight download, dataset download, training, frontend/API
change or GitHub operation is part of Phase 6. The only deliverable is this
readiness/design document; verification measurements/baselines are local ignored
working artifacts under backend/.tmp. All original data and previous phases are
preserved.

## 8. Completion verification

On 2026-11-15 the full existing backend suite passed: **74 tests**, with the one
previously documented Starlette/httpx deprecation warning. The frontend production
build passed. Tests used their existing temporary fixtures; no new tests, training
code or dependencies were added for this documentation phase.

SHA-256 comparison confirmed all **65 pre-existing source, document, downloader,
manifest and report baseline files** remained byte-identical. A final independent
check reconfirmed all **40 raw videos, 200 extracted frames and 204 face crops**
against their asset hashes. PyTorch remains absent. All existing unassigned split
values, previous reports and original media were preserved. No acquisition or
training was performed. Phase 6 is complete and stops at documentation.
