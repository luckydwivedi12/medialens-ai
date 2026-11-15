# Phase 3: offline dataset audit

This CLI inspects local media already obtained under the dataset's official access
procedure. It never downloads, trains, runs inference, repairs metadata, changes
source media, or extracts frame files. Temporary generated media in tests are
software fixtures only. Phase 1 API/frontend and Phase 2 documentation are preserved.

## Run

From the repository root in PowerShell:

```powershell
.\backend\.venv\Scripts\python.exe backend/scripts/audit_dataset.py --dataset-root "D:\MediaLensData\raw\faceforensics" --dataset-name "faceforensics"
```

With a populated Phase 2 manifest and selected extensions:

```powershell
.\backend\.venv\Scripts\python.exe backend/scripts/audit_dataset.py --dataset-root "D:\MediaLensData\raw\faceforensics" --dataset-name "faceforensics" --manifest "D:\MediaLensData\manifests\faceforensics.csv" --output-dir "D:\MediaLensData\reports" --extensions jpg,png mp4 --sample-frames 3
```

`--extensions` accepts comma-separated or space-separated subsets of JPG, JPEG,
PNG, WEBP, MP4, MOV, AVI and MKV, case-insensitively. Other files are ignored.
`--sample-frames` accepts 1-32, default 3, with positions spread over the reported
frame count. `--help` does not touch datasets or create reports. Progress goes to
stderr; final report paths go to stdout.

Exit codes: 0 = completed with passed or warning status; 1 = completed with audit
errors; 2 = invalid options or inability to finish/write reports. Automation must
inspect overall_status and issues: exit 0 with warning is not a clean audit.

## Responsibilities

- `app/data/models.py`: inventory, validation, and issue records.
- `inventory.py`: incremental directory discovery, stat metadata, streaming SHA-256
  with 1 MiB chunks, and changed-during-hashing detection. Linked roots are refused;
  nested symlinks/junctions are skipped and reported rather than followed.
- `image_validation.py`: Pillow structure verification and pixel decoding; reports
  original dimensions and color mode. Decompression-bomb warnings are errors.
- `video_validation.py`: OpenCV open/metadata checks and a bounded sample of frames;
  capture resources are released in a finally block, including failure paths.
- `duplicate_detection.py`: exact byte-hash groups, stable hash-based group IDs,
  paths and individual byte sizes. No automatic removal.
- `leakage_check.py`: manifest schema/row validation, source and checksum split
  overlap, and transitive original/derivative component overlap.
- `report.py`: coordinates validation and streams inventory rows to CSV.
- `scripts/audit_dataset.py`: argument parsing and exit/status presentation.

Files are hashed sequentially; only one media file/frame is decoded at a time.
The scanner yields records incrementally and does not read entire files for hashing.
Duplicate and leakage indexes, checksum sets, and issues retain metadata proportional
to file/manifest count; this is not a constant-memory solution for millions of rows.
Very large catalogs may eventually need an on-disk index. No images/videos are kept
in those indexes. Keep the corpus stable during an audit; this is not a filesystem
snapshot and cannot guarantee consistency under concurrent edits.

## Reports and preservation

Defaults:

```text
D:\MediaLensData\reports\dataset_audit.json
D:\MediaLensData\reports\dataset_inventory.csv
```

Every report is opened in exclusive-create mode. If a default name already exists,
a UTC timestamp and UUID suffix produce a new filename. No overwrite option exists.
The JSON records the exact CSV path. Concurrent audits cannot overwrite each other's
files. Output inside the dataset root is rejected to keep dataset directories
read-only. Only the report directory is created when required; the scanner never
creates a missing dataset root. A failed/interrupted write can leave an incomplete
new report; it never replaces an older report. Consumers should require a complete,
parseable JSON report before interpreting its CSV as a completed audit.

The JSON includes dataset name/root, UTC timestamps, scope/configuration, file
count, total discovered media bytes, extension distribution, valid/invalid counts,
traversal errors, exact duplicates, missing metadata, leakage violations, severity
counts, and overall status. Directory/link traversal failures are counted separately
from media files. Unreadable entries may have unknown sizes; totals are observed
bytes, not guaranteed whole-folder storage. CSV includes each discovered media
path, extension, kind, size, UTC mtime, SHA-256, validation status, JSON metadata and
errors. Discovery failures also produce diagnostic CSV rows.

Any error makes status `failed`. With no errors, warnings make it `warning`.
Only absence of both yields `passed`. An empty directory, a header-only manifest,
or no manifest always produces a warning. This tool never certifies authenticity,
dataset rights, training readiness, or absence of every form of leakage.

## Manifest rules and leakage scope

Use the unchanged [Phase 2 header](dataset_manifest_template.csv) and
[field definitions](dataset_strategy.md). All 12 column names are required; unique
extra columns are allowed but not interpreted. Values use the documented canonical
lowercase vocabulary; invalid values are reported, not normalized or repaired.

- Required IDs, label, split and checksum missing: errors. Missing public source,
  access date or terms reference: provenance warnings. Nonempty dates must be
  YYYY-MM-DD. `access_approved` must equal `true`; this is checked as a metadata
  assertion, not verified against the provider.
- Labels are real, fake or unknown. Unknown is a warning and is ineligible for
  supervised training. Invalid labels are errors. Split values are train,
  validation, test, independent_test, development or quarantine.
- Frame indices are blank for video rows or non-negative integers for frames.
  Checksums must be `sha256:` plus 64 lowercase hex digits.
- Source identity is scoped by dataset_name and dataset_version; use namespaced
  source IDs as the Phase 2 strategy requires. Every manifest row is checked;
  dataset_name differing from the requested dataset is an error.
- Cross-split equal source IDs, equal checksums, and transitive original/derivative
  components are errors. Sibling fakes sharing an absent original are grouped too.
  Real originals must point to themselves. A violation reports every affected row
  in its group; one record may legitimately participate in multiple violation types.
- Inventory checksums must have manifest records. Video-level manifest checksums
  must occur in the selected inventory; missing matches are errors. With extension
  filtering, supply a manifest for that same audit scope. Frame-only manifests
  cannot be used to match raw video bytes.

The minimal manifest contains no file paths, donor graph, upstream recording IDs,
face tracks or companion asset index. Consequently this phase does not establish
whether a frame references the right parent, whether two IDs describe the same
upstream video, or whether donors/identities leak across datasets. Missing originals
cannot be inferred from filenames. Supply a complete manifest: non-represented
relationships cannot be checked. Only checksum identity is checked across differing
dataset/version namespaces. Do not infer full Phase 2 split purity from this audit.

## Dependencies and offline verification

Added Pillow 13.0.0, NumPy 2.5.4, and opencv-python 5.0.0.93 to backend requirements.
Existing pins are unchanged. Cached wheel metadata supports Python 3.12 and Windows
x64: Pillow/NumPy cp312 and OpenCV cp37-abi3. OpenCV requires NumPy >=2 on this
Python version. The standard OpenCV distribution was already cached locally;
opencv-python-headless was not. Only VideoCapture is used in production audit code;
GUI functions are unused. Never install both OpenCV distributions in one environment.

These packages were installed from local cached wheels using pip --no-index;
no network requests were made. No PyTorch or other model libraries were installed.
The copied wheels and preservation baseline are under ignored backend/.tmp and are
not application artifacts or dataset/model downloads.

Verification commands from backend:

```powershell
.\.venv\Scripts\python.exe -m pytest
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe -c "from app.data import inventory, image_validation, video_validation, duplicate_detection, leakage_check, report"
.\.venv\Scripts\python.exe scripts/audit_dataset.py --help
```

The suite uses pytest temporary files for inventory errors, tiny Pillow images,
corrupt media, an MJPG AVI, duplicates, split leakage, malformed metadata, missing
checksum references, safe report creation, CLI help/options, and empty audits.
Near-duplicate detection, complete video corruption scans, face detection, extraction,
decoding of every animated-image frame, label truth verification, donor/identity
graph audits and dataset-specific metadata importers remain future work. FPS-derived
duration is an estimate, especially for variable-frame-rate videos. Codec support
and seeking reliability depend on OpenCV's local backend. A hung native decoder has
no subprocess timeout in this version. Never use final-test audit results to tune a
model or preprocessing recipe.

## Verified completion (2026-11-15)

- Complete backend suite: **40 passed** (37 new audit tests and 3 existing API
  tests); one existing Starlette/httpx deprecation warning. Tests ran outside the
  Windows sandbox because the existing TestClient requests previously stalled there.
- Frontend production build: passed; no frontend source was changed.
- pip check: no broken requirements. All eight audit modules import successfully
  from backend; CLI help succeeds from both backend and the repository root.
- An empty temporary directory audit created dataset_audit.json and
  dataset_inventory.csv under D:\MediaLensData\reports. Result: zero media files,
  zero errors, two warnings (empty inventory and no manifest). The temporary root
  was removed after verification; the report remains as a smoke-test record, not
  a real FaceForensics audit.
- SHA-256 comparison against the saved local baseline: 22 original files,
  including existing application/test source and Phase 2 documents, remained
  byte-identical. Only backend/requirements.txt changed among baseline files.
  There is no initialized Git repository, so no Git status/diff was available.
- Added eight app/data modules, scripts/audit_dataset.py, seven tests/data files
  including its package initializer, and this document. Existing requirement pins
  and requirements-dev.txt remain unchanged; only the three audit dependencies
  were appended to requirements.txt.

Stop after Phase 3; no acquisition or preprocessing phase is started here.
