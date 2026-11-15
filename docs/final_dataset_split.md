# Phase 9: dataset quality control and split design

Assessment date: **2026-11-15, Asia/Calcutta**. The dataset now has a versioned,
validated lineage split and a development-only visual review package. All crop
labels remain inherited video labels or provisional region candidates. This
phase does not approve training labels or establish classifier accuracy.

## Evidence and scope

Read AGENTS.md, expanded_dataset_report.md, mask_assisted_face_curation.md and
training_data_readiness.md. Reopened the actual expanded video, extraction,
detection, mask-validation, mask-association and crop-pair CSVs and the verified
Phase 7/8 reports. Counts below were recomputed from individual records.

The current cohort contains 400 videos, 2,000 frames, 2,015 saved crops and 93
explicit detection failures. The current preprocessing run is
`20261115T093847254224Z_cac1903c48ac4bd1b63d30b993c4fc6c`.
The Phase 8 curation run is
`20261115T100655195781Z_1260d214dd024346b5b706c114ad84f8`.
Historical images are included in the lineage/duplicate audit but are not added
to current sample counts as independent observations.

## Explicit count reconciliation

| Population | Count | Reconciliation |
| --- | ---: | --- |
| Detection and failure records | 2,108 | 2,015 crop records + 93 zero-face records |
| Saved crops | 2,015 | 1,002 original-video + 1,013 fake-video crops |
| Fake-video crops | 1,013 | 865 mask-supported candidates + 93 insufficient-support target crops + 55 nonoverlapping bystander candidates |
| Mask-supported candidates | 865 | 854 matched region pairs + 11 unmatched candidates |
| Zero-face records | 93 | 51 original-video frames + 42 manipulated-video frames |
| Manipulated frame records | 1,000 | 865 uniquely supported + 93 insufficient support + 42 zero-face |
| Frames with multiple detections | 87 | 43 original + 44 manipulated frames |
| Crops in those multiple-face frames | 195 | Overlaps the crop populations above; not an additional population |
| Multiple-mask-overlap ambiguous records | 0 | Does not imply zero uncertainty |
| Matched original/manipulated region pairs | 854 | 1,708 crops: 854 original + 854 manipulated candidates |
| Original crops outside the paired subset | 148 | 1,002 minus 854; retained for review |
| All crops outside the paired subset | 307 | 148 original + 11 unmatched fake candidates + 93 insufficient support + 55 bystander candidates |

Every unmatched candidate has an actual original frame and a saved `no_face`
record with error **“No faces above confidence threshold.”** The saved detector
threshold is 0.9. All eleven passed Phase 8 same-index, timestamp and background
alignment checks, and have empty original-detection IoU lists. Thus the gap
represents absent saved original detections, not mask/video alignment failures
or failed matching among existing original boxes. A detector failure does not
mean a face is absent from the image. No detector was rerun at a lower threshold.

| Manipulated video | Target original | Frame index | Partition | Reason |
| --- | --- | ---: | --- | --- |
| 892_112 | 892 | 155 | train | Original has no saved detection above threshold |
| 263_284 | 263 | 270 | validation | Same |
| 284_263 | 284 | 338 | validation | Same |
| 452_550 | 452 | 60 | test | Same; pixels not reviewed in Phase 9 |
| 452_550 | 452 | 181 | test | Same; pixels not reviewed in Phase 9 |
| 452_550 | 452 | 424 | test | Same; pixels not reviewed in Phase 9 |
| 706_479 | 706 | 98 | test | Same; pixels not reviewed in Phase 9 |
| 706_479 | 706 | 164 | test | Same; pixels not reviewed in Phase 9 |
| 786_819 | 786 | 364 | train | Original has no saved detection above threshold |
| 786_819 | 786 | 468 | train | Same |
| 948_965 | 948 | 130 | train | Same |

The package contains `count_reconciliation.csv` and `unmatched_candidates.csv`.
All eleven unmatched records, including five locked-test records, retain their
crop/frame paths, failure evidence, group and partition. Nothing is discarded.

## Official split and recording research

The official [dataset documentation](https://github.com/ondyari/FaceForensics/blob/master/dataset/README.md)
links to sequence-pair split files and a sequence-to-YouTube conversion table.
Rechecked [train.json](https://github.com/ondyari/FaceForensics/blob/master/dataset/splits/train.json),
[val.json](https://github.com/ondyari/FaceForensics/blob/master/dataset/splits/val.json)
and [test.json](https://github.com/ondyari/FaceForensics/blob/master/dataset/splits/test.json).
The frozen Phase 8 bundle was checked against immutable objects from the official
GitHub repository; the downloaded bytes were independently checked with Git's
blob-hash algorithm and compared with the frozen parsed pairs.

| Official file | Verified Git blob SHA |
| --- | --- |
| train.json | `979240f9091412ad17b998ab7b06899e54c3b771` |
| val.json | `731b584efea371a81a55c59b3200c2b1320d72f4` |
| test.json | `854b8019a5e279473ac41671d0946b8476920d6b` |
| conversion_dict.json | `84d0b60a22b7ac64b0f996d9d02ab5a72b1cc9ed` |

The [official conversion metadata](https://github.com/ondyari/FaceForensics/blob/master/dataset/conversion_dict.json)
contains YouTube recording identifiers and extraction indices, not person
identity annotations. The frozen raw content and its Git/SHA-256 evidence are
included as `official_recording_metadata.json` in the final package.

All 200 original sequence IDs and all 100 target/source pairings map to official
partitions. No sequence has conflicting official partition membership, no cohort
pairing is absent, and no component spans official partitions.

## Leakage graph and the additional recording link

First rebuild the undirected target/source graph: original sequence IDs are
nodes; each manipulation joins target and donor. This reproduces the **100
pairing components** from Phase 8, with 200 nodes and 100 unique edges.

Then add edges between sequences sharing the same official upstream YouTube
recording ID. There are **198 distinct recording IDs** among the 200 originals:

- `nmfPtUu2nSM`: sequences **614 and 615**, previously in the separate pairing
  groups 614–616 and 615–687. They merge into **{614, 615, 616, 687}**.
- `yjsNEL7DQuY`: sequences **942 and 943**, already in one pairing component.

The stricter, recording-aware graph therefore has **99 components**: one
four-sequence component and 98 two-sequence components. Both groups joined by
the additional recording link already belong to official training. There are
zero shared recordings crossing the final partitions; all video/crop assignments
remain identical to the official sequence mapping. This discovery reduces the
number of lineage units without moving the final test set.

Checks cover source and target references, original/manipulated derivatives,
matched crop relationships, and historical derivative manifests. SHA-256 file
hashes are checked across videos, masks, current/historical frames and crops.
Decoded-image hashes include dimensions and BGR pixels and detect identical
images even if their PNG encodings differ. This is exact duplicate detection,
not near-duplicate or biometric identity clustering.

## Split methodology and actual counts

**Use verified official membership**, normalized from `val` to `validation`.
Do not reshuffle components to approximate 70/15/15 when compatible official
metadata exists. The reproducible fallback uses canonical component IDs,
Python's local `random.Random(42)`, and 70/15/15 allocation; it is tested but was
not used for this cohort. No frame or crop is split independently.

| Population | Train | Validation | Locked test | Total |
| --- | ---: | ---: | ---: | ---: |
| Recording-aware leakage components | **66** | **17** | **16** | **99** |
| Original target/source pairing components | 67 | 17 | 16 | 100 |
| Videos | 268 | 68 | 64 | 400 |
| Extracted frames | 1,340 | 340 | 320 | 2,000 |
| All preserved crops | **1,368** | **361** | **286** | **2,015** |
| Original-video crop labels | 681 | 180 | 141 | 1,002 |
| Fake-video crop labels | 687 | 181 | 145 | 1,013 |
| Detection failure records | 39 | 14 | 40 | 93 |
| Mask-supported manipulated candidates | 590 | 152 | 123 | 865 |
| Matched region pairs | **586** | **150** | **118** | **854** |
| Paired-subset crop candidates | **1,172** | **300** | **236** | **1,708** |
| Paired-subset original/fake video labels | 586 / 586 | 150 / 150 | 118 / 118 | 854 / 854 |
| Unmatched manipulated candidates | 4 | 2 | 5 | 11 |
| Other preserved crops outside paired subset | 196 | 61 | 50 | 307 |

Both inherited video labels are represented in every partition. The paired view
is balanced by construction, not a validated face-ground-truth benchmark.
`label` and `inherited_video_label` retain video provenance; `label_scope`
explicitly rejects interpreting them as face truth. `selection_status`,
`paired_subset_candidate` and `pair_id` identify the provisional paired view.
Every row remains `manual_review_required=True` and `training_approved=False`.
All uncertain crops and zero-face failures still receive their lineage partition
and remain present, with no invented matched counterpart.

Assignments exist only in new versioned manifests. Old manifests still contain
their original unassigned values. A video-lineage map and an asset inventory
extend the same partition policy to previous-run derivatives. Historical images
are not to be concatenated with current images as extra independent samples.

## Deterministic visual review package

Only training and validation records enter selection, sorting, confidence ranking
or rendering. No test pixels are rendered in a Phase 9 review sheet. Deterministic
sample-ID SHA-256 ordering selects one record per recording-aware component first,
then fills remaining positions. This spreads examples across lineage units;
it does not make the examples a statistically representative random survey.

| Contact sheet | Entries | Selection |
| --- | ---: | --- |
| authentic_crops_01.png | 8 | Crops from original videos |
| manipulated_candidates_01.png | 8 | Mask-supported manipulated candidates |
| matched_pairs_01.png | 8 | Manipulated crop left, matched original crop right |
| high_localization_confidence_01.png | 8 | Highest saved localization confidence, with component diversity |
| low_localization_confidence_01.png | 8 | Lowest saved localization confidence, with component diversity |
| multiple_faces_01.png | 8 | Full frames with existing detections overlaid |
| unmatched_candidates_01.png | 6 | All development unmatched candidates; original frame shown because no crop exists |
| failures_and_insufficient_01.png | 8 | Four zero-face frames and four insufficient-support examples |

The 62 index entries cover 51 distinct primary sample records; categories may
overlap, and pair cards also display a counterpart. Every card displays sample
ID, original/manipulated video IDs, inherited leakage-group ID, label scope,
frame index, localization confidence and mask-overlap score. Unavailable values
are explicitly N/A. The CSV additionally records the recording-aware component,
input paths, counterpart metadata, every detection in the displayed parent frame,
sheet coordinates and pending review status. A related manipulated ID on an
original-only example is a lineage reference unless a verified pair is recorded.

Gold boxes indicate a mask-supported target candidate; green boxes indicate
other detections. The images are copied into new contact sheets with thumbnail
scaling and annotations. Original crops, frames and videos are not changed.
“Low confidence” refers to localization rank within the development pool, not a
manipulation probability. No new curation, image-quality or classifier threshold
is selected from these sheets.

Visual QA verifies readability, layout and correspondence to recorded source
images; it does not approve every face label. Development examples visibly
include detector misses despite visible faces, very small or screen-face
detections, broadcast graphics, low-detail crops and manipulation artifacts.
These are reasons for later recorded review, not automatic deletion. All review
decisions remain `pending_human_review`.

## Final-test lock and independence limits

`test_lock.json` fixes final-test component IDs, split-manifest checksum,
assignment digest and per-asset checksums. The development review renderer and
its independent checker reject test records and test image paths. The lock is
a versioned policy and integrity control, not an operating-system access barrier.
Do not use test examples for curation-threshold changes, augmentation/normalization
fitting, checkpoint selection or classifier calibration. Future final evaluation
requires a separately authorized evaluation phase.

**The lock is prospective.** Phase 8 computed overlap sensitivity statistics
over the entire cohort, including records now assigned to test; earlier phases
also performed full-cohort QC. No classifier trained on these records, and Phase
9 does not change the existing thresholds, but this is not a claim that test data
has never informed data preparation. A pristine final benchmark would require a
separately reserved cohort and a frozen preparation protocol.

- **Video-lineage independence:** known target/donor links and official shared
  recording IDs stay within one split. Exact byte/pixel duplicates do too. This
  is the independence actually audited here; reuploads and near duplicates are
  not exhaustively resolved by distinct recording IDs.
- **Person-identity independence:** different sequences or recording IDs may
  depict the same person. Official split and conversion files do not supply a
  person-identity disjointness guarantee. No identity model, biometric matching
  or person-name identification was run. Identity independence remains unverified.
- **Dataset independence:** all current examples come from one dataset, one
  manipulation method and c23 compression. These partitions do not measure
  independent-dataset generalization. No external evaluation dataset was acquired.

## Implementation and reproduction

Code is separated into `backend/app/data/split_design.py` (graph and split
validation), `dataset_quality.py` (reconciliation/provenance/integrity), and
`visual_review.py` (review selection and contact sheets). It uses existing
OpenCV, Pillow and standard-library dependencies. No API, inference or frontend
source changes are required.

From `backend`, run:

```powershell
.venv\Scripts\python.exe scripts\prepare_dataset_split.py `
  --phase8-report D:\MediaLensData\reports\mask_face_association_report_verified_20261115T102014703702Z_b5fb4bc7cd294cec90bb68fb40bf5d94.json `
  --storage-root D:\MediaLensData `
  --verify-official-live
```

For offline reproduction, replace `--verify-official-live` with
`--recording-metadata <final-package>\official_recording_metadata.json`.
The pinned official split bundle is read through the verified Phase 8 report;
the offline recording bundle is checked against its original Git blob hash.
Each invocation creates a unique timestamp/UUID version and opens every output
exclusively. Reproduction must preserve assignment membership and deterministic
review selections; timestamps, paths and preservation-baseline sizes can differ.

Run the independent verifier with:

```powershell
.venv\Scripts\python.exe scripts\verify_dataset_split.py --report <final-phase9-report.json>
```

It rechecks preserved files and output checksums, recomputes graph and official
mapping, validates every detection/failure record, checks matched-pair lineage,
verifies locked test membership and proves no test image path entered the review
index. If design validation fails, the producer keeps actual assignments
unassigned and retains proposed memberships and failure evidence for review.

Required verification commands are `.venv\Scripts\python.exe -m pytest` from
`backend` and `npm run build` from `frontend`. No PyTorch, model training,
new datasets, original-media edits, uncertain-sample deletion, manifest/report
overwrite or GitHub push is part of this phase.

## Final artifacts and verification

Final version: `phase9_20261115T104516317959Z_8f2f4d7e21b84ff795f566fef7452c03`.
Use this recording-aware version for later work. Earlier uniquely named Phase 9
iterations remain preserved; they are not additional independent datasets.

Review package root:
`D:\MediaLensData\reports\phase9_visual_review\phase9_20261115T104516317959Z_8f2f4d7e21b84ff795f566fef7452c03`.
Contact sheets are in its `contact_sheets` subdirectory, with index
`contact_sheets\index.csv`. The root also contains count reconciliation,
all eleven unmatched candidates, component assignments, official recording
metadata, preservation baseline, test lock and output checksums.

New versioned manifests:

- `D:\MediaLensData\manifests\dataset_split_manifest_phase9_20261115T104516317959Z_8f2f4d7e21b84ff795f566fef7452c03.csv`
  — 2,108 current detection/failure records, including 2,015 crops.
- `D:\MediaLensData\manifests\video_lineage_split_phase9_20261115T104516317959Z_8f2f4d7e21b84ff795f566fef7452c03.csv`
  — all 400 source-video assignments and recording-aware component IDs.
- `D:\MediaLensData\manifests\split_asset_inventory_phase9_20261115T104516317959Z_8f2f4d7e21b84ff795f566fef7452c03.csv`
  — 5,019 physical assets: 4,615 current and 404 historical, each with lineage,
  partition and file checksum; 4,419 decoded images also have pixel hashes.

Measured report:
`D:\MediaLensData\reports\dataset_split_report_phase9_20261115T104516317959Z_8f2f4d7e21b84ff795f566fef7452c03.json`.
Independent verification is saved in the package as
`independent_verification_20261115T104655391887Z_4bed2885ed1f452c8eb631f2615b46a9.json`.

| Verification | Actual result |
| --- | --- |
| Recording-aware components crossing splits | 0 |
| Source/target sequence IDs crossing components | 0 |
| Original/manipulated and matched crop relationships crossing splits | 0 |
| Shared upstream recordings crossing splits | 0 |
| Exact file checksums crossing splits | 0 |
| Exact decoded-image checksums crossing splits | 0 |
| Expected historical/current byte-duplicate groups | 404, all within splits |
| Current detection/failure records preserved | 2,108 / 2,108 |
| Unmatched candidates preserved | 11 / 11 |
| Pre-existing external files preserved byte-for-byte in final run | 5,119 / 5,119 |
| Distinct files independently checksum verified | 5,138 |
| Locked test assets (historical runs also audited) | 702 |
| Phase 9 rendered test images / test review rows | 0 / 0 |
| Full backend test suite | **106 passed**, one existing Starlette/httpx deprecation warning |
| Frontend production build | **Passed** |

Canonical assignment SHA-256:
`107e5daaf4576ce6df0c06f8a94148cfbdc4dbc9928aadbd8a8056497f82e5f6`.
The extra shared-recording edge changes component IDs/counts but not a single
video's official partition; the final-test membership is unchanged.

Sixteen new tests cover official precedence, transitive graph merges,
recording aliases within/across official partitions, deterministic seed-42
fallback, invalid references, file/pixel duplicate leakage, historical duplicate
handling, derivative and matched-pair tampering, unverified alignment, test
exclusion before review ranking, confidence ordering, count reconciliation and
preservation of unmatched original-detection failures. The full suite used normal
external temporary-directory access. No dependency was installed.

Phase 9 is complete. Human annotation decisions and any future training or final
evaluation remain separate work; no later phase has started.
