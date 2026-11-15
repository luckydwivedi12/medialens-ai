# Phase 8: mask-assisted face curation and split audit

Run date: 2026-11-15 (Asia/Calcutta). This phase curates **region-supported
candidates** from existing crops. It does not establish face authenticity labels,
train a model or allocate train/validation/test splits.

## Measured results and saved outputs

| Measurement | Actual result |
| --- | ---: |
| Expected / downloaded / valid masks | 200 / 200 / 200 |
| Masks matching video dimensions, count, FPS and sampled timestamps | 200 |
| Missing / invalid masks | 0 / 0 |
| Fully decoded mask frames | 103,288 |
| Mask/video positions with histograms and timestamp checks | 1,400 |
| Exact binary (0/255), equal-channel sampled masks | 1,400 / 1,400 |
| Empty foreground masks at inspected positions | 0 |
| Existing manipulated frames re-decoded and pixel/checksum verified | 1,000 |
| Unique mask-supported face candidates | 865 |
| Insufficient-support frames | 93 |
| Zero-face manipulated frames | 42 |
| Multiple-overlap ambiguous frames / ambiguous face records | 0 / 0 |
| No-overlap frames with detections | 0 |
| Nonoverlapping bystander crop candidates | 55 |
| Manipulated frames with multiple detections | 44 |
| Candidate frames passing original temporal/content checks | 865 |
| Original/manipulated matched region crop pairs | 854 |
| Selected candidates with no original detection | 11 |
| Frames unresolved before original pairing | 135 |
| Detection/failure records preserved | 2,108 / 2,108 |
| Crop records preserved | 2,015 / 2,015 |
| Zero-face records preserved, real / fake | 93, comprising 51 / 42 |
| Actual sequence-relationship connected components | 100 |
| Official split conflicts / missing pairings / unknown IDs | 0 / 0 / 0 |

There were no curation exceptions. All 865 selected candidates passed the
original same-index/timestamp/background gates. Eleven original frames had no
saved face detection, so their crop relationships remain unresolved. The other
135 frames comprise 93 insufficient-support cases and 42 zero-face failures;
original pairing was not attempted for them. Zero multiple-overlap ambiguity
does not imply zero uncertainty: those unresolved cases remain visible.

Full-cohort threshold sensitivity (all other gates held constant): overlap 0.25
selects 958 candidates; 0.50 selects 865; 0.75 selects eight. Each setting retains
42 zero-face records. At 0.75 another 950 frames have insufficient support.
This substantial sensitivity is another reason to retain the annotations and
scores rather than treat the selected subset as face-level ground truth.

Final visual review covered paired single-face and multi-face examples,
an insufficient-support frame and an original-detection failure. In the latter,
the original face is visibly present but lacks a saved detection; it was
correctly left unresolved. The two reviewed contact sheets, with checksums,
are preserved alongside the reports and referenced by the final verified report.

All three output CSVs use suffix
`20261115T100655195781Z_1260d214dd024346b5b706c114ad84f8`:

- `D:\MediaLensData\manifests\mask_validation_manifest_20261115T100655195781Z_1260d214dd024346b5b706c114ad84f8.csv`
- `D:\MediaLensData\manifests\mask_assisted_face_manifest_20261115T100655195781Z_1260d214dd024346b5b706c114ad84f8.csv`
- `D:\MediaLensData\manifests\face_pair_manifest_20261115T100655195781Z_1260d214dd024346b5b706c114ad84f8.csv`

Final report, combining original run results, validation aggregates, inspection
evidence, test outcomes and independent verification:
`D:\MediaLensData\reports\mask_face_association_report_verified_20261115T102014703702Z_b5fb4bc7cd294cec90bb68fb40bf5d94.json`.

The original run report is also preserved:
`D:\MediaLensData\reports\mask_face_association_report_20261115T100655195781Z_1260d214dd024346b5b706c114ad84f8.json`.
The independent verification report is
`D:\MediaLensData\reports\mask_curation_verification_20261115T101831061437Z_3d2c14f27ff54842b8b21799224f48eb.json`.
It verified 5,040 distinct referenced files against recorded SHA-256 values,
including every current source video/frame/crop/mask and all 461 historical
baseline files. Detection/failure closure and the 1,000 pair/failure rows passed;
every split remains unassigned. No existing crop was deleted or overwritten.

Frozen official split evidence:
`D:\MediaLensData\reports\official_splits_20261115T100655083872Z_bdbaba4ab15f422ca57dd517acc6ba3f.json`,
SHA-256 `fea5390dcd1c202378b00803c46cdcc5fdb43acc48c934733e091261a3d92dca`.

Final required verification: **90 backend tests passed** with the existing
Starlette/httpx deprecation warning, and **frontend production build passed**.
The backend suite ran with normal external temporary-directory access after a
sandboxed attempt could not create pytest fixtures. The 15 new tests cover
matching, metadata/temporal alignment, binary/grayscale interpretation, face
intersection, unique and ambiguous selections, original-region matching,
transitive graph components and official conflicts, missing masks, content
misalignment, decoded counts, exclusive outputs, and end-to-end preservation
of missing-mask/zero-face failures without invented crop pairs.

No new dependencies, training, classifier evaluation, split allocation, frontend
source changes, original-media edits, crop deletion, commit or push occurred.
Phase 8 is complete; later phases have not been started.

## Official annotation research and acquisition

Read the official [dataset documentation](https://github.com/ondyari/FaceForensics/blob/master/dataset/README.md)
and [DeepFakes documentation](https://github.com/ondyari/FaceForensics/blob/master/dataset/DeepFakes/README.md).
The former specifies binary masks, target_sequence_source_sequence filenames,
and a rectangular Deepfakes editing region. The latter explains that the region
marks where Poisson editing was applied. These are region annotations, not
pixel-perfect manipulation ground truth.

Inspected the unmodified approved script at
`D:\MediaLensData\tools\faceforensics_download_v4.py`. Its EU2 mask URL is
`http://kaldir.vc.in.tum.de/faceforensics/v3/manipulated_sequences/Deepfakes/masks/videos/`.
The live `v3//misc/filelist.json` expands each pair in both directions before
truncating to 200. Its selected filename set exactly matched the expanded
manifest's 200 manipulated videos. No mask directory existed before acquisition.
Free disk space was 98,165,686,272 bytes; the resumed preflight also enforced a
20 GiB reserve. The 200 completed masks occupy 8,383,109 logical bytes.

The script always requests Terms of Use confirmation. Acquisition paused until
the user explicitly accepted the terms in the Phase 8 chat. A wrapper then ran
the supplied script unchanged with these arguments:

```powershell
python D:\MediaLensData\tools\faceforensics_download_v4.py D:\MediaLensData\raw\faceforensics --dataset Deepfakes --type masks --num_videos 200 --server EU2
```

The server timed out after 41 masks. The same command resumed with a 45-second
socket timeout and checked subsequent filelist responses against a frozen copy.
The official script skipped existing files. Its failed temporary download files
were preserved and excluded from mask counts; no mask or original media was
overwritten. The completed download contains exactly 200 MP4 mask files in
`D:\MediaLensData\raw\faceforensics\manipulated_sequences\Deepfakes\masks\videos`.
The unique `mask_download_preflight_*.json` in the external reports directory
records the verified selection and complete filelist.

## Inputs and implementation

The expanded video manifest is
`D:\MediaLensData\manifests\faceforensics_100_groups_manifest.csv`.
The frame and detection manifests both use preprocessing run
`20261115T093847254224Z_cac1903c48ac4bd1b63d30b993c4fc6c`.
Inputs contain 400 videos, 2,000 extracted frames, 2,015 crops and 93 zero-face
failure records. All splits are unassigned.

Implementation is isolated in `backend/app/data/mask_association.py`
(mask interpretation, overlap ranking, matching, graph algorithms) and
`backend/app/data/mask_curation.py` (read-only media validation and manifests).
The CLI is `backend/scripts/curate_mask_faces.py`; an independent integrity
checker is `backend/scripts/verify_mask_curation.py`. No new dependencies,
detectors, classifier inference, API changes or frontend edits are involved.

Derived manifests must refer to the exact source-manifest checksum and matching
source-video checksums. Mixed preprocessing runs, duplicate frames, foreign
video IDs and assigned splits are rejected. Filename target/source IDs must
agree with the video manifest. Each manipulated video and original counterpart
is checksum checked before use. Output creation is exclusive and uses a timestamp
and UUID suffix; historical outputs remain in place.

## Mask and temporal validation

For every expected manipulated video, record the mask path and SHA-256 or an
explicit missing/invalid result. Require identical positive frame count,
dimensions and FPS. Fully decode each mask, require a consistent resolution,
and count decoded frames independently of the mask header. Source-video total
counts remain decoder metadata; this is not a full source-video corruption scan.

At each existing sampled frame index plus the first, middle and last mask
positions, record decoded positions, mask/video timestamps, the complete 256-bin
grayscale histogram, observed pixel values, exact binary/grayscale status,
channel equality and connected foreground regions. Require decoder positions
to match the requested index and timestamps to agree within 0.1 frame (at least
1 ms). Missing decoder timestamps do not pass through an FPS-based estimate.

At each sampled manipulated frame, re-decode the video and require exact pixels
and checksums to match the stored extracted frame. All foreground components
contribute to overlap; nothing is silently removed as noise. Pixel values above
127 are foreground for decoded 0–255 masks. An explicitly detected 0/1 binary
encoding uses values above zero. Grayscale values are recorded, not silently
declared binary. Masks are never rescaled to force agreement.

Index/timestamp evidence establishes sampled decoder alignment, not an
independent proof that every annotation is correctly drawn or semantically timed.
Visual overlay inspection provides additional sampled spatial support.

## Face association and threshold inspection

Every existing detection retains its box, confidence, landmarks, crop path,
checksum, parent video/frame identifiers and failure record. Detector confidence
is a localization score, not an authenticity score.
The retained `label` column is the inherited **video** label. In a fake video it
still says `fake` even for a bystander record; it must not be used as a face label.
Use `association_status`, `face_role` and `face_label_status` to interpret the
curation evidence. No new binary face labels are assigned.

For each YuNet box, measure foreground intersection divided by full rasterized
face-box area, and intersection divided by total foreground area (mask coverage).
Clip intersection to the image; do not shrink the denominator for an off-image
face. Rank candidates by face overlap, then coverage. A unique candidate requires
**only one box with any foreground intersection**, face overlap at least **0.50**
and mask coverage at least **0.05**. Two intersecting faces are ambiguous even
when one intersects by only a few pixels. No overlap, insufficient support and
zero-face cases stay explicit; no uncertain face is forced into a selection.
Nonoverlapping detections are bystander candidates, without an authenticity label.

Inspection before the full run covered 180 available manipulated-frame samples:
155 unique candidates, 23 insufficient-support cases and two zero-face frames
at the chosen threshold. Positive face-overlap quantiles (min, 10%, 25%, median,
75%, 90%, max) were 0.426, 0.493, 0.532, 0.582, 0.627, 0.663 and 0.759.
An 11-image overlay contact sheet was visually reviewed: eight single-face
examples, a presenter plus a screen face, and two detector failures. The masks
covered inner face regions; in the multi-face example only the presenter overlapped.
The 0.50 threshold conservatively leaves borderline inner-face masks unresolved.
The 0.05 coverage floor prevents accepting a tiny face relative to a large region;
it is a heuristic safeguard, not a calibrated statistical operating point.

The full report also records sensitivity at overlap thresholds 0.25, 0.50 and
0.75. These are inspection heuristics, not thresholds optimized against labeled
face-level truth. Candidate and bystander assignments remain reviewable.

## Authentic counterparts

Take the target ID from the manipulated filename, not the donor ID. Require
matching dimensions, FPS and total frame-count metadata before considering an
original counterpart. Use the same exact extracted frame index and require
decoder timestamps to match. Re-decode the original and compare its pixels and
checksum to the saved original frame.

For content support, dilate the mask by 31 pixels and compare grayscale image
content outside it. Require at least 20% remaining background, grayscale standard
deviation of at least 5, correlation at least 0.98 and mean absolute difference
at most 8 intensity levels. This excludes flat or strongly mismatched backgrounds.
The gates are conservative heuristics, not an independent temporal registration
algorithm; static scenes and upstream timing differences remain limitations.

Once those checks pass, find original detections with box IoU at least 0.50.
Only a unique qualifying original detection receives a crop relationship.
Store original/fake crop paths, checksums, IoU values and alignment evidence.
Any missing frame, timing/metadata/content mismatch, absent face or multiple
qualifying originals yields an explicit unresolved/ambiguous record. These are
spatiotemporal region-pair candidates, not biometric identity assertions.

## Leakage graph and official splits

Build an undirected graph with original sequence IDs as nodes and each
target/source relationship as an edge. Reciprocal directions share one edge.
Traverse the graph for connected components, allowing transitive relationships
to merge any overlapping filename-derived groups.

The measured graph has **200 nodes, 100 unique edges and 100 connected components**.
All components have two sequence IDs. There are no missing original references,
shared IDs across components, or merged reciprocal groups in this cohort.
This result is computed from the graph, not assumed from the 100 group labels.

Compared against the official [train](https://github.com/ondyari/FaceForensics/blob/master/dataset/splits/train.json),
[validation](https://github.com/ondyari/FaceForensics/blob/master/dataset/splits/val.json)
and [test](https://github.com/ondyari/FaceForensics/blob/master/dataset/splits/test.json)
files. Raw-content endpoints reset the connection; the official GitHub contents
API supplied the same paths with Git blob IDs. The frozen bundle and its SHA-256
are referenced in the association report.

**All 100 pairings are compatible with the official files**: 67 components map
to train, 17 to validation and 16 to test. There are zero unknown sequence IDs,
missing official pairings, component partition conflicts or sequence IDs appearing
in multiple official partitions. This is an audit comparison only: existing
manifests and all new records remain `split=unassigned`.
Identity independence between different original sequence IDs and independence
of upstream recordings remain unverified without supporting metadata.

## Reproduction

From `backend`, use the existing virtual environment and explicit input run:

```powershell
.venv\Scripts\python.exe scripts\curate_mask_faces.py `
  --manifest D:\MediaLensData\manifests\faceforensics_100_groups_manifest.csv `
  --frames D:\MediaLensData\manifests\frame_extraction_20261115T093847254224Z_cac1903c48ac4bd1b63d30b993c4fc6c.csv `
  --faces D:\MediaLensData\manifests\face_detection_20261115T093847254224Z_cac1903c48ac4bd1b63d30b993c4fc6c.csv `
  --dataset-root D:\MediaLensData\raw\faceforensics `
  --storage-root D:\MediaLensData `
  --thresholds-reviewed
```

For an offline repeat, pass `--official-splits` with the frozen bundle path
recorded in the report. Configurable options are `--foreground`,
`--minimum-face-overlap`, `--minimum-mask-coverage` and `--original-iou`.
Only pass `--thresholds-reviewed` after documenting inspection; omitting it
keeps every selection flagged for review. Downloads are separate from this CLI;
it never accepts dataset terms or downloads media itself.

Run `scripts\verify_mask_curation.py --report <new-report-path> --dataset-root
D:\MediaLensData\raw\faceforensics` with the same virtual environment. It
independently rehashes every source video, saved frame, crop, acquired mask and
input manifest, verifies detection/failure closure, checks that every fake frame
has a pair/failure record, and rejects matched pairs lacking alignment evidence
or any allocated split. Its report also uses exclusive unique creation.
Pass `--preservation-baseline` with the earlier `preservation_baseline_*.json`
to additionally recheck the historical 461-file baseline.

Required checks: `.venv\Scripts\python.exe -m pytest` from `backend`, and
`npm run build` from `frontend`. No model training, PyTorch installation,
additional ML dependencies, split assignment, crop deletion, commit or push
is part of Phase 8.
