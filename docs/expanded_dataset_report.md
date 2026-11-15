# Phase 7: expanded FaceForensics++ preprocessing report

Completed on **2026-11-15 (Asia/Calcutta)**. This is a bounded development
cohort with dataset video labels, not an authenticity or classifier benchmark.
All results below are measured unless explicitly called estimates.

## Acquisition and preflight

Read AGENTS.md and all four requested dataset/preprocessing/readiness documents.
Inspected the supplied official downloader at
`D:\MediaLensData\tools\faceforensics_download_v4.py`; it remains unchanged.
Its SHA-256 is `3d0aa3a186ec2f210056cdfcc5fba97d6f77f7872cfafd1754063954683a47ba`.
Access evidence and terms acceptance remain the owner's prior attestation,
preserved in every source record; the email was not independently authenticated.

Retrieved the live EU2 filelist at the script's configured endpoint:
`http://kaldir.vc.in.tum.de/faceforensics/v3//misc/filelist.json`.
The frozen complete filelist SHA-256 is `7099a119c0992751a8fd58fda33569ccb01442d8d13aad7fea9e69dc1d2510f4`.
The first 100 pairs contain 200 distinct original IDs and 200 distinct reciprocal
Deepfakes IDs. All ten previous groups occur in that prefix; their paths,
relationships and SHA-256 values match. No original has conflicting pair membership.
The complete first-100-pair list and exact 400 expected paths are recorded in
`D:\MediaLensData\reports\expansion_preflight_20261115T093421925121Z_4b2e3e538bea4c5d9ecfc096e741b4f1.json`.

For originals, the downloader flattens the pairs before truncating to 200 entries.
For Deepfakes videos it emits A_B and B_A before the same truncation. Thus both
commands select exactly these 100 groups. A local execution wrapper invoked the
unmodified supplied script with dataset=original or Deepfakes, compression=c23,
type=videos, num_videos=200 and server=EU2. Every filelist request was checked
against the frozen list before returning it to the script; a mismatch would stop
the run. No alternate mirror, manipulation, model weight or full-dataset download
was used. Existing video paths were skipped by the official script.

The official server reset connections or timed out repeatedly. Downloads resumed
using the same script and selection, with a 45-second socket timeout on later
attempts. Both jobs ultimately exited successfully. All download logs and the
complete official filelist were copied to unique names in the external reports
directory. Eight empty staging files left by failed attempts were preserved at
`D:\MediaLensData\tmp\download_staging_20261115T093845988557Z_605594ad88cf4cebac77321380863740\tmpau04bl9w`
(the other seven files are in that same directory). None were counted as videos;
the raw dataset tree now contains exactly 400 MP4 files and no other files.

Free D: space at preflight: **100,521,627,648 bytes**.
Estimated incremental growth was **2,355,449,755
bytes**, using nine times the old raw-video bytes plus ten times the old image
bytes, since all 400 videos receive a new processing run. This was an estimate,
with an 8 GB growth allowance and 20 GiB free-space reserve.

## Actual inventory, manifest and storage

| Measurement | Result |
| --- | ---: |
| Original c23 videos / real records | 200 |
| Deepfakes c23 videos / fake records | 200 |
| Total unique video files / manifest records | 400 / 400 |
| Reciprocal leakage groups | 100 |
| Valid / invalid videos | 400 / 0 |
| Zero-byte videos | 0 |
| Traversal errors / conflicting video IDs | 0 / 0 |
| Missing original references / incomplete reciprocal groups | 0 / 0 |
| Exact SHA-256 video duplicate groups | 0 |
| Raw video storage | 756,324,058 bytes |
| Added raw video storage above old sample | 678,122,133 bytes |
| New processing run, including images, CSVs and both summary copies | 1,661,634,229 bytes |
| Raw videos plus new processing run | 2,417,958,287 bytes |

Storage figures are logical file lengths, not filesystem allocation. They exclude
the preserved old processing run, download logs, audit/case/verification reports
and filesystem overhead. Empty staging files consume zero logical payload bytes.

The existing dataset audit pipeline ran once for the acquired inventory and again
with the expanded manifest. It recorded SHA-256, dimensions, FPS, frame count,
estimated duration and sampled decode results for every video. **Readability means
successful metadata probing and three sampled-frame decodes per video, not a full
corruption scan of every frame.** The later extraction checks five further uniform
sample positions per video.

Expanded manifest: `D:\MediaLensData\manifests\faceforensics_100_groups_manifest.csv`.
SHA-256: `sha256:df269ea6d9c87c6b4f3186d251a79b1ee81882a1914eb133c28a59dc10a2a016`.
It retains the 20-column source schema, target/base and donor IDs, access evidence,
video hashes and deterministic `ffpp-c23-pair-A-B` IDs using sorted original IDs.
Every group has two real originals and both reciprocal fake videos. Every path,
file size, hash and relationship was verified against actual files. All records
remain `split=unassigned`; `frame_index` is blank at video level. The descriptive
purpose is `expanded_preprocessing_development`. The existing dataset namespace
and variant remain unchanged for compatibility; this manifest is a superset and
must not be concatenated with the original sample as independent observations.

Final audit: `D:\MediaLensData\reports\dataset_audit_20261115T093812815516Z_9a0a15abde3941df84777ae5cf6a0dbe.json`.
Inventory with per-video metadata: `D:\MediaLensData\reports\dataset_inventory_20261115T093613706923Z_ac19a90dd9f744d1a058a0f304c0a998.csv`.
Result: **zero errors and one intentional `unassigned_split` warning**.
There are no missing-metadata or reported leakage violations. Without allocated
splits, this is not a demonstrated train/validation/test leakage guarantee.

## Frame extraction and YuNet detection

Reused the existing Phase 5 pipeline, five distinct indices at approximately
10%, 30%, 50%, 70% and 90% of each video's reported frame count. Variable-frame-rate
timing remains approximate. No resizing, alignment or target-face selection was
added. Sequential CPU processing used confidence=0.9 and crop margin=0.2 per side.

Run ID: `20261115T093847254224Z_cac1903c48ac4bd1b63d30b993c4fc6c`.
Frames: `D:\MediaLensData\processed\frames\20261115T093847254224Z_cac1903c48ac4bd1b63d30b993c4fc6c`.
Crops: `D:\MediaLensData\processed\face_crops\20261115T093847254224Z_cac1903c48ac4bd1b63d30b993c4fc6c`.

The existing `face_detection_yunet_2026may.onnx` was SHA-256 verified before use:
`ebafce4e3c118d6554634be5c27ab333b4c047a9a8c3faf1d7cf93101c22f0f0`.
Its provenance remains OpenCV Zoo commit
`48b10fa505aeb51fbd127fc60a5c660040e823b7`, MIT model license, native CPU
ENGINE_NEW (2). No replacement detector or ONNX Runtime fallback was used.
The existing OpenCV target warning remains documented in Phase 5.

| Measurement | Actual result |
| --- | ---: |
| Videos processed | 400 |
| Frames extracted / failed | 2000 / 0 |
| Inherited real / fake frame labels | 1000 / 1000 |
| Faces detected / crops saved | 2015 / 2015 |
| Inherited real / fake crop labels | 1002 / 1013 |
| Frames with multiple saved faces | 87 |
| Zero-face sampled frames | 93 |
| Processing errors / warnings | 0 / 93 |
| Exact byte duplicate groups within new frames/crops | 0 |
| Exact decoded-pixel duplicate groups within new frames/crops | 0 |
| New assets matching preserved earlier-run bytes | 404 |
| Pipeline runtime | 415.31 seconds |

The earlier-run matches are expected reprocessing of the existing sample,
not additional independent evidence. Duplicate comparisons inside the new run
were performed separately for full frames and crops. Near-duplicate clustering,
identity and upstream-recording independence were not established.

Frame manifest: `D:\MediaLensData\manifests\frame_extraction_20261115T093847254224Z_cac1903c48ac4bd1b63d30b993c4fc6c.csv`.
Face manifest: `D:\MediaLensData\manifests\face_detection_20261115T093847254224Z_cac1903c48ac4bd1b63d30b993c4fc6c.csv`.
Processing report: `D:\MediaLensData\reports\processing_report_20261115T093847254224Z_cac1903c48ac4bd1b63d30b993c4fc6c.json`.
Multiple-face case rows: `D:\MediaLensData\reports\multiple_faces_20261115T093847254224Z_cac1903c48ac4bd1b63d30b993c4fc6c.csv`.
Zero-face case rows: `D:\MediaLensData\reports\zero_faces_20261115T093847254224Z_cac1903c48ac4bd1b63d30b993c4fc6c.csv` (header-only if no cases).
Independent integrity report: `D:\MediaLensData\reports\expansion_verification_20261115T093847254224Z_cac1903c48ac4bd1b63d30b993c4fc6c.json`.

Both derived manifests preserve video ID, target and donor, inherited label,
unassigned split, leakage group, frame index, timestamp/method, source/video/frame
checksums, detector confidence, coordinates, landmarks and model provenance.
Every new image was independently decoded and checksum verified. The verification
also checked expected sample indices, metadata inheritance, frame dimensions,
square crop coordinates and bounds, parent links and exact disk/manifest closure.
Zero-face events remain explicit face-manifest rows and have a separate case CSV;
multiple detections are retained individually in their separate case CSV.

## Label uncertainty, preservation and verification

**A fake video label is not verified ground truth for every face crop.** Bystanders,
detector false positives and missed faces remain possible. All crop rows explicitly
retain `inherited_video_label_not_face_ground_truth`. Confidence measures face
localization, not manipulation or authenticity. Manual target/track or authorized
annotation review is still required before selecting training examples. Zero-face
counts are detector outcomes, not a measurement of its false-negative rate.

The 100 groups reflect known filename-derived target/donor relationships, not
proven independent identities, YouTube recordings or cross-dataset sources. This
one-method c23 cohort cannot support claims about other methods or real-world
accuracy. No classifier accuracy was measured or claimed.

All **461 pre-existing external files** were
SHA-256 rechecked and remain byte-identical, including the original 40 videos,
40-video manifest, 200 frames, 204 crops, model/provenance and earlier reports.
New manifests and reports used exclusive creation and unique run/report names.
The original 40-video manifest was not overwritten.

The manifest builder now accepts `--expected-groups` (default 10 preserves the
earlier workflow); `--expected-groups 100` enforces 400 records with the same
relationship/hash checks. A new 100-group fixture verifies balanced complete
groups and rejects a cross-pair donor change without a reciprocal. No application,
API, inference, frontend source or dependency changes were needed.

Final required verification: **75 backend tests passed**, with the pre-existing
Starlette/httpx deprecation warning; **frontend production build passed**.
The canonical backend command was `.venv\Scripts\python.exe -m pytest` from
backend, run outside the sandbox with its normal external temporary directory.
Earlier sandbox/temp-directory attempts could not satisfy fixture access or the
pipeline's repository-output guard; the canonical full suite passed. The frontend
command was `npm run build` from frontend.

No packages were installed, PyTorch was not installed, no training or classifier
inference occurred, no train/validation/test splits were assigned, and no GitHub
push was made. **Phase 7 stops here.**
