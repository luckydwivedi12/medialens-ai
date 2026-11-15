# Phase 5: sampled video frames and YuNet face localization

Completed and verified on **2026-11-15**. This is preprocessing of the approved
40-video FaceForensics++ c23 Deepfakes smoke-test sample. YuNet localizes faces;
it does not predict fake/real status or establish authenticity. All dataset labels
are inherited from the source-video manifest, and all splits remain unassigned.

## Implementation

New files:

- `backend/app/data/frame_extraction.py`: distinct uniform sample indices,
  sequential per-video processing, explicit extraction failures, resource release,
  and exclusive lossless PNG saving with decode/pixel validation.
- `backend/app/data/face_detection.py`: pinned-model/hash/provenance validation,
  dynamic-input smoke checks and FaceDetectorYN localization.
- `backend/app/data/face_cropping.py`: finite-box validation, margins, boundary
  handling and square native-resolution crops.
- `backend/app/data/preprocessing.py`: source-manifest validation, orchestration,
  metadata inheritance, external run directories, manifests and summary reports.
- `backend/scripts/extract_video_frames.py`: dedicated CPU command-line entry point.
- Four corresponding test modules: `test_frame_extraction.py`,
  `test_face_detection.py`, `test_face_cropping.py`, `test_preprocessing.py`.
- This documentation file.

No existing application/API, frontend, tests, dependency pins, or previous phase
documents were changed. No dependency installations were necessary.

## Model provenance and compatibility

Official source: [OpenCV Zoo YuNet directory](https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet).
The actual downloaded repository commit is
`48b10fa505aeb51fbd127fc60a5c660040e823b7`.

| Property | Verified value |
| --- | --- |
| Model | face_detection_yunet_2026may.onnx |
| Bytes | 229,738 |
| SHA-256 | ebafce4e3c118d6554634be5c27ab333b4c047a9a8c3faf1d7cf93101c22f0f0 |
| License | MIT, as specified by the model directory's LICENSE |
| Storage | D:\MediaLensData\models\yunet\face_detection_yunet_2026may.onnx |
| Python package | opencv-python 5.0.0.93 |
| Runtime-reported OpenCV | 5.0.0 |
| FaceDetectorYN | Available |
| Production engine | ENGINE_NEW (2), CPU only |
| Confidence threshold | 0.9 |
| NMS IoU threshold / top_k | 0.3 / 5000 |

The [commit-pinned model download](https://media.githubusercontent.com/media/opencv/opencv_zoo/48b10fa505aeb51fbd127fc60a5c660040e823b7/models/face_detection_yunet/face_detection_yunet_2026may.onnx)
was checked against the pinned Git LFS object's declared SHA-256 and byte size.
Existing files were checked before use and were not overwritten. The
[model license](https://raw.githubusercontent.com/opencv/opencv_zoo/48b10fa505aeb51fbd127fc60a5c660040e823b7/models/face_detection_yunet/LICENSE)
and [README](https://raw.githubusercontent.com/opencv/opencv_zoo/48b10fa505aeb51fbd127fc60a5c660040e823b7/models/face_detection_yunet/README.md)
were also saved beside the model with commit-qualified filenames. Keep the license
notice with any permitted model redistribution. Dataset terms remain separate.

The official YuNet README describes dynamic height/width for the 2026may model and
the OpenCV 5.x ONNX Runtime engine (`OPENCV_FORCE_DNN_ENGINE=4`). This installed
wheel reports **ONNX Runtime: NO**. Requesting engine 4 did not use ONNX Runtime:
OpenCV warned and fell back to ENGINE_AUTO. Initial compatibility results, including
that missing-runtime build flag, were retained at:

`D:\MediaLensData\models\yunet\compatibility_20261115T084812012784Z_d75d8ac01dcc47eba2991d993b209523.json`.

We did not treat fallback success as proof that ORT is available.
[OpenCV's engine documentation](https://docs.opencv.org/5.0/main_modules/dnn_engine_selection.html)
also documents the native ENGINE_NEW as CPU-only and supporting dynamic shapes.
That engine was then selected explicitly, tested at 320x320, 854x480 and 1280x720,
and the exact same YuNet model was checked on two original-resolution sample
images in memory. The production CLI sets `OPENCV_FORCE_DNN_ENGINE=2` before
importing OpenCV; its constructor verifies forward passes at two dimensions.
All 200 production frames subsequently completed localization successfully.
The production report records engine 2, the native-runtime note and model hash.

OpenCV emits `Targets are not supported by the new graph engine for now` while
FaceDetectorYN initializes its target. The explicitly selected engine is CPU-only;
this is not evidence of GPU or ORT execution. The warning is documented rather
than hidden. No alternative face detector, model version or Haar cascade was used.
If a future build cannot load or run this pinned model, the CLI stops before
processing rather than switching detectors or downloading replacements.

## Sampling and validation

For N reported video frames, use unique sorted indices
`min(N-1, floor(N*f))` for f in 0.1, 0.3, 0.5, 0.7 and 0.9. Very short videos yield
fewer than five samples, never repeated indices. A 100-frame video yields
10, 30, 50, 70 and 90. This approximates the requested temporal fractions; it is
not exact timestamp-based sampling for variable-frame-rate footage.

One VideoCapture is open at a time and is released in a finally block, including
early generator closure. Each sample checks seek/read success, decoded position,
positive dimensions and uint8 BGR image shape. A decoded frame position must agree
with the requested index. Decoder POS_MSEC is recorded when usable; otherwise the
timestamp is estimated as frame_index/FPS with its method explicitly recorded.

Original frame dimensions are retained. PNG outputs are opened exclusively, fully
decoded and checked for pixel equality with the in-memory sample. SHA-256 is then
recorded for every saved asset. Processing holds one sampled frame and a crop at
a time, not the whole video. PNG validation temporarily allocates decoded copies;
metadata and report memory scale with the manifest and issue counts.

Failures produce explicit manifest records and processing-error entries. A video
that cannot provide valid metadata produces one video-level extraction-failure
event; `frames_failed` counts recorded extraction/save failures, not five invented
failed frame indices. No-frame and no-face events are distinct. No-face detection
creates a face-manifest row referencing the retained frame, with status no_face,
an explanatory error and a summary warning. No such cases occurred in this run.

## Detection and crop policy

FaceDetectorYN receives the sampled frame at its original dimensions. Every
returned detection meeting the threshold after YuNet's configured NMS/top_k is
recorded separately, including its floating-point x, y, width, height, confidence
and five landmark coordinate pairs in YuNet output order. There is no selection
of a presumed primary or manipulated face, no face tracking and no alignment.

Default margin is 0.2 per side. Desired square side length is
`ceil(max(box_width, box_height) * (1 + 2*margin))`. Shift the square into the image
bounds; cap its side to the shorter image dimension. Coordinates are integer,
right/bottom-exclusive. Nonfinite, nonpositive or fully out-of-image boxes fail
explicitly. At extreme boundaries, a square cannot always include every part of
a very large box plus its desired margin; original face coordinates and final
crop coordinates are both retained for review.

Crop pixels are copied at native resolution and saved independently as PNGs.
No original frame or crop is resized. Detection scores are localization confidence,
not deepfake probabilities. A crop inherited from a fake-labeled video may show an
unmanipulated bystander; the manifest explicitly flags this label limitation.

## Output layout and manifest schema

Run ID: `20261115T085602795863Z_ab2d90f7dbc34410880fa34482f99d1d`.

```text
D:\MediaLensData\
  processed\frames\<run_id>\<video_id>_frame_<index>.png
  processed\face_crops\<run_id>\<video_id>_frame_<index>_face_<number>.png
  manifests\frame_extraction_<run_id>.csv
  manifests\face_detection_<run_id>.csv
  manifests\processing_summary_<run_id>.json
  reports\processing_report_<run_id>.json
```

Directories and filenames use a UTC timestamp plus UUID. Existing run paths are
rejected; assets, CSVs and JSONs use exclusive creation. Raw media and repository
storage are rejected as processed-output locations. An interrupted run may leave
partial new outputs; only a completed summary establishes a finished run. Old
outputs are neither deleted nor overwritten.

Both CSVs retain the original manifest's 20 columns, with frame_index set to the
decoded source-video index and checksum set to the saved asset's SHA-256 (blank
on failures/no-face events). The source manifest's relative_path still refers to
the raw video; frame_path and crop_path are the derived-asset locations.

| Metadata | Definition |
| --- | --- |
| dataset_name, dataset_version, source_url, access_date, access_approved, terms_reference, access_evidence | Copied source provenance; user-attested access evidence is preserved |
| source_video_id, video_id | Exact parent video, not the face donor |
| original_video_id, target_video_id, donor_video_id, manipulation_method | Copied original relationships; never inferred from detection |
| label, leakage_group_id, split, sample_purpose | Copied video-level annotation/group; split remains unassigned |
| run_id, asset_type | Processing run and frame/face_detection record type |
| source_video_checksum, source_manifest_sha256 | Immutable parent-video and source-manifest hashes |
| frame_index, timestamp_seconds, timestamp_method | Decoded video position and decoder/estimated timestamp provenance |
| frame_path, frame_checksum | Absolute saved full-frame path and its SHA-256 |
| width, height | Original sampled frame dimensions, including on face rows |
| detection_index, detection_confidence | Zero-based within-frame detection number and localization score |
| face_coordinates | JSON [x,y,width,height] in original-frame pixels |
| face_landmarks | JSON array of five [x,y] pairs in detector output order |
| crop_coordinates | JSON [left,top,right,bottom] of square in original frame |
| crop_path, checksum | Saved crop path/hash for successful face rows; frame hash for successful frame rows |
| detector_name, detector_model_version, detector_model_sha256 | YuNet, model filename and raw 64-digit model SHA-256 |
| face_label_status | inherited_video_label_not_face_ground_truth |
| status, error | Extracted/crop_saved, no_face or explicit failure stage and explanation |

Inapplicable values such as face coordinates on a frame record are blank. Failures
retain source metadata and a reason instead of disappearing. The summary/report
includes runtime configuration, model provenance, sampling, counts, video-level
results, storage and every processing error. The identical summary is stored in
manifests and reports for the requested two uses.

## Actual processing results

Run completed at **08:56:38 UTC / 14:26:38 IST**, 2026-11-15.

| Metric | Result |
| --- | ---: |
| Videos processed | 40 |
| Frames extracted | 200 (5 per video) |
| Frame failures | 0 |
| Faces detected / crops saved | 204 / 204 |
| Frames / videos without detected faces | 0 / 0 |
| Average faces per extracted frame | 1.02 |
| Processing duration | 36.20 seconds |
| Image bytes | 165,163,243 |
| Total new run-output bytes | 165,828,737 (~158.15 MiB) |
| Processing errors / warnings | 0 / 0 |
| Processing status | passed |

The total includes images, two CSVs and both JSON copies, excluding the separately
stored model/provenance/license files. A passed processing status means these
operations succeeded, not that the data is ready for training or authenticated.

Frames: 100 inherited real labels and 100 inherited fake labels. Crops: 102
inherited real-video labels and 102 inherited fake-video labels. Multiple faces
were returned for some sampled frames; every detection was kept independently.

Actual report files (append the run ID above):

- `D:\MediaLensData\manifests\frame_extraction_<run_id>.csv` — 200 records.
- `D:\MediaLensData\manifests\face_detection_<run_id>.csv` — 204 records.
- `D:\MediaLensData\manifests\processing_summary_<run_id>.json`.
- `D:\MediaLensData\reports\processing_report_<run_id>.json`.

## Reproduction and validation

From the repository root, using the already verified local model/provenance:

```powershell
.\backend\.venv\Scripts\python.exe backend/scripts/extract_video_frames.py --manifest "D:\MediaLensData\manifests\faceforensics_sample_manifest.csv" --dataset-root "D:\MediaLensData\raw\faceforensics" --storage-root "D:\MediaLensData" --model "D:\MediaLensData\models\yunet\face_detection_yunet_2026may.onnx" --model-provenance "D:\MediaLensData\models\yunet\compatibility_20261115T084812012784Z_d75d8ac01dcc47eba2991d993b209523.json" --confidence 0.9 --margin 0.2
```

Each execution creates a new run; it does not reacquire the model or any dataset.
CLI exit codes: 0 for completed passed/warning runs, 1 for a completed processing
report with errors, 2 for configuration/model/manifest or fatal I/O failure.
Inspect the report as well as the exit code.

Before processing real media, **74 backend tests passed**, including all 48 prior
tests. New tests use only tiny generated MJPG videos, images and controlled detector
outputs: no restricted dataset assets or model downloads are automated fixtures.
Coverage includes distinct sampling indices, invalid video failures, resource
release, square/clipped crops, invalid boxes, all detections, confidence filtering,
landmarks, no-face events, metadata/groups, integrity checks and safe output reuse.
The frontend production build, new CLI help and pip check also passed. The sole
test warning remains the previously documented Starlette/httpx deprecation.

After the production run, all 200 frame and 204 crop files were independently
decoded and checksum-verified. Crop dimensions and image-boundary coordinates,
five landmarks, confidence thresholds, per-video distinct indices and metadata
inheritance were checked. All ten leakage groups and unassigned splits were
preserved. All 40 raw video hashes still match the original manifest. All 50
pre-existing source/document/manifest/report baseline files remained byte-identical.
The downloaded model hash matches upstream. No API/frontend or dependency change
was made, and no old report was overwritten.

## Remaining limitations

- Only ten filename-derived leakage groups and one manipulation/compression
  variant are represented. This remains a smoke test, not a meaningful benchmark.
- Five samples cannot describe every video event; frame-index timing approximates
  duration fractions for variable-frame-rate footage.
- Detector false positives/negatives and crop quality have not been quantified.
  No-face failures are implemented and tested, although absent in this sample run.
- Multiple detected faces are not linked into tracks or matched to manipulated
  targets. Inherited video labels must not be treated as verified per-face truth.
- Square crops have variable dimensions and do not guarantee identity alignment
  or complete context at image boundaries. No training transforms were added.
- Near-duplicate, identity and cross-dataset overlap checks remain future work.
- The native engine works with the exact pinned model in this wheel. ORT itself
  remains unavailable; changing the wheel or runtime requires new compatibility
  checks. Installing Python onnxruntime alone would not establish that OpenCV was
  compiled with its integration.
- This is a single-process pipeline; native decoder/model calls have no hard
  subprocess timeout. A stable source corpus is required during execution.

No deepfake classifier, authenticity prediction, training, additional dataset
download, split assignment, API change or GitHub operation was performed.
Stop after Phase 5.
