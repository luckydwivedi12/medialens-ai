# MediaLens AI: Phase 2 dataset strategy

Sources checked: **2026-11-15 (Asia/Calcutta)**. This is a research and preparation
plan, not evidence that dataset access, decoding, extraction, or training works.
No access forms were submitted. No data, model weights, packages, or synthetic
training examples were acquired. Phase 1 application code and dependencies are
outside the scope of changes.

## Scope and decision

Detect manipulated human faces in images and sampled video frames, with future
video-level aggregation. Audio detection and fully generated scenes without
manipulated faces are excluded. Dataset labels describe the dataset's annotation;
an eventual model score must never be presented as proof of authenticity.

Recommended plan, conditional on access and terms:

1. **Primary training: FaceForensics++ c23**, beginning with a bounded subset of
   complete source groups and all four core manipulation methods. It matches the
   scope, offers lineage and masks, and has a manageable compressed footprint.
2. **Validation: audited official FaceForensics++ splits**, with validation-only
   checkpoint/threshold selection and a locked within-dataset test set. Apply the
   stronger source-group audit below before accepting the official assignments.
3. **Independent evaluation: Celeb-DF v2**, completely excluded from training,
   calibration, and preprocessing parameter selection. Its synthesis differences
   offer a useful generalization test. Approval and the scope of derived-data
   permission must be resolved first.
4. **Preprocessing development: a small labeled DFDC training sample**, if the
   official Kaggle sample is still available under acceptable terms. Start with a
   proposed 20-40-video subset of complete available original/derivative groups;
   this is a project budget, not an official dataset size. It exercises video and
   metadata handling without full DFDC. Keep it development-only. A bounded
   approved preview subset is a fallback; do not acquire either in this phase.

On the i5-1235U, 16 GB RAM, Iris Xe machine without CUDA, storage fit does not imply
practical full-scale training. Future work should measure CPU decoding speed and
bounded frame sampling before selecting a model or committing to training time.
No ML dependencies or preprocessing implementation are introduced here.

## Evidence register

All links below were checked on the date above. V = verified in readable official
documentation; E = project estimate/proposal; U = unresolved or inaccessible.
Public pages can describe historical procedures without proving current server
availability. Recheck terms and endpoints immediately before future acquisition.

| ID | Official evidence | Result |
| --- | --- | --- |
| F1 | [FaceForensics repository](https://github.com/ondyari/FaceForensics) | V: overview and core methods |
| F2 | [Dataset documentation](https://github.com/ondyari/FaceForensics/blob/master/dataset/README.md) | V: paths, filenames, metadata, compression sizes, approved-script procedure |
| F3 | [Official access form and dataset terms](https://docs.google.com/forms/d/e/1FAIpQLSdRRR3L5zAv6tQ_CKxmK4W96tAab_pfBu2EKAgQbeDVhmXagg/viewform) | V: terms read; not submitted |
| F4 | [Paper](https://arxiv.org/html/1901.08971v3) and [split directory](https://github.com/ondyari/FaceForensics/tree/master/dataset/splits) | V: split policy; train.json inspected as paired IDs |
| F5 | [Repository software license](https://github.com/ondyari/FaceForensics/blob/master/LICENSE) | V: software permission, distinct from dataset terms |
| D1 | [Meta dataset page](https://ai.meta.com/datasets/dfdc/) | V: approximate release sizes and AWS prerequisites |
| D2 | [Full DFDC paper, section 3.5](https://arxiv.org/html/2006.07397v4) | V: exact training counts and evaluation partitions |
| D3 | [Preview paper](https://arxiv.org/html/1910.08854v2) | V: preview counts, identity metadata; inconsistent counts noted below |
| D4 | [Kaggle data](https://www.kaggle.com/competitions/deepfake-detection-challenge/data) and [rules](https://www.kaggle.com/competitions/deepfake-detection-challenge/rules) | U: pages returned no readable body; sample bytes, schema, current access and terms not verified |
| C1 | [Celeb-DF repository](https://github.com/yuezunli/celeb-deepfakeforensics) and [paper](https://arxiv.org/html/1909.12962v4) | V: core counts and synthesis; public v2 directory has images, not the requested test-list text |
| C2 | [Official Celeb-DF access/terms form](https://docs.google.com/forms/d/e/1FAIpQLScoXint8ndZXyJi2Rcy4MvDHkkZLyBFKN43lTeyiG88wrG0rA/viewform) | V: terms and version choice read; not submitted |
| R1 | [DeeperForensics repository](https://github.com/EndlessSora/DeeperForensics-1.0) | V: research purpose and paper overview |
| R2 | [DeeperForensics dataset documentation](https://github.com/EndlessSora/DeeperForensics-1.0/blob/master/dataset/README.md) | V: released counts, storage, access, target overlap and split warning |
| R3 | [Linked terms PDF](https://github.com/EndlessSora/DeeperForensics-1.0/blob/master/dataset/Terms_of_Use.pdf) | U: PDF content not readable through the research tool; full clauses not verified |

## Dataset comparison

Counts refer to videos, not independent frames. Compression copies and transformed
variants must not be counted as independent source examples. The table uses core
or specified release scopes rather than combining incompatible versions.

| Dataset / scope | Total videos | Real | Fake | Manipulations | Disk footprint | Access | Usage restrictions | Training | Validation | Independent testing |
| --- | ---: | ---: | ---: | --- | --- | --- | --- | --- | --- | --- |
| FaceForensics++ core [F1,F4] | 5,000 | 1,000 | 4,000 (1,000/method) | Deepfakes, Face2Face, FaceSwap, NeuralTextures | c23 ~10 GB; c40 ~2 GB; raw ~500 GB [F2] | Form approval, then supplied script | Non-commercial research/education [F3] | Primary | Audited official groups | Within-dataset test; not independent of its training |
| DFDC full training [D2] | 119,154 | 19,154 (subtraction) | 100,000 | DFAE, MM/NN, NTH, FSGAN families | Current official archive bytes U | Meta AWS route / Kaggle route to verify | Full accepted terms U | Future only; excessive initial scope | Identity-disjoint partitions in paper | Future, if held out entirely |
| DFDC preview [D1,D3] | ~5k; paper discrepancy below | U in preview paper checked | U in preview paper checked | Two methods A/B | Archive bytes U | Request and terms; Meta lists AWS prerequisites | Terms acceptance required; clauses U | Not primary | Only grouped/official | Possible; reserve development subset separately |
| Celeb-DF v2 core [C1] | 6,229 | 590 | 5,639 | Improved DeepFake face swapping | Official byte size U | Form approval, academic email requested | Non-commercial research; strict copying/derived-data clauses [C2] | Technically possible; excluded here | Excluded here | Recommended, conditional on permission |
| DeeperForensics-1.0 released files [R2] | 59,475 | 48,475 actor source videos, NOT detector real controls | 11,000 | DF-VAE and perturbation variants | ~284 GB; >=300 GB recommended | Terms then approved form | Non-commercial research [R1]; full PDF clauses U | Deferred | Future audited splits | Not automatically independent of FF++ targets |

### FaceForensics++ research and access

The core originals come from 977 YouTube videos; masks support localization.
Do not conflate the separately distributed Google/Jigsaw DeepFakeDetection
collection with the four-method core. [F1,F2]

F2 documents original IDs 000-999, target_source manipulated filenames,
conversion_dict.json, and video information. These enable lineage reconstruction.
After a later authorized form submission and approval, use only the provided
script, selecting c23 videos and required methods; omit weights and bulk images.
F2 additionally estimates source YouTube downloads at 38.5 GB and PNGs at ~2 TB.
These are published approximations, not measured local requirements. [F2]

The F3 agreement limits use to non-commercial research/education, requires
associates to accept terms before access, includes responsibility and termination
clauses, and requests dataset citations. The software license does not grant
unrestricted rights over the media. Commercial product/model use needs separate
clarification. [F3,F5]

F4 describes 720/140/140 original-video train/validation/test partitions. All
associated core manipulations inherit the audited assignment. These are not
frame counts and not a claim of identity-disjointness. The paper's hidden benchmark
is separate from a local held-out split. [F4]

### DFDC research and access

Meta describes a ~5k preview with two algorithms and a ~124k full release with
eight; these approximations do not replace the precise training scope in D2.
The Meta route lists an AWS account, IAM user/access keys and account ID as
prerequisites. This documents the route, not successful access today. Do not ask
for or store credentials in the project. [D1]

D2 reports 10-second training clips, 4,000 public validation clips (2,000 fake),
and 10,000 private-test clips (half fake). Only the DFDC-like part of that private
test is described as released; do not assume all test media/labels are available.
Paid participants consented to likeness modification. [D2]

D3 reports 5,214 in its table but 4,464 train + 780 test = 5,244 in its narrative.
It documents dataset.json with swapped/target identity, split, and augmentation
information. Resolve counts against the approved release inventory rather than
silently choosing a number. Its two method names do not disclose implementations.
[D3]

For the desired small Kaggle training sample, exact count, class distribution,
download size and current metadata schema remain U because D4 was unreadable.
Do not certify common third-party sample counts. The future import contract
should require filename, REAL/FAKE label, original reference for fakes, and release
partition, inspecting whether these appear in metadata.json or another file.
This is a proposed contract, not a verified Kaggle schema. Never use unlabeled
competition test videos for supervised validation.

Future access: sign in on the official competition page, inspect current rules,
record acceptance, and select only the labeled training sample if offered. If
unavailable, stop that route; do not substitute an unapproved mirror or full
corpus. Frame processing is suitable in principle because video-level labels and
source relationships can accompany sampled frames. A fake-video label alone does
not prove every face in every frame was manipulated; record track/mask validity.

### Celeb-DF v2 research and access

C1 verifies 590 originals from 59 celebrities and 5,639 improved-synthesis
DeepFakes. The 6,229 core total is arithmetic. Common expanded-release claims
(300 extra real videos, 6,529 total, 518-video test list) were not verified in
readable official material here: retain them as U until approved v2 metadata
confirms them. No official disk size was established. [C1]

Use the repository-linked form, select v2, provide the requested affiliation,
purpose, academic email and agreement only in a future authorized access step;
the authors send a link after acceptance. Terms restrict non-commercial research,
require citation and forbid further copying/publication/distribution, including
derived data. The copying language is broad: obtain clarification on local frame
and crop processing before doing it, and on any model/artifact release. [C2]

Independent evaluation is a project recommendation based on different synthesis
and source collection, not a guarantee of statistical independence. Audit
YouTube/source overlap with FF++ first. Use the supplied official evaluation list
if confirmed on receipt; otherwise predeclare the full eligible v2 cohort before
scoring. Do not select a favorable subset after seeing predictions.

### DeeperForensics-1.0 research and access

Keep deferred; do not acquire. R1's paper overview rounds to 60,000 videos and
17.6 million frames. Use the exact released inventory scope in the table. DF-VAE,
seven perturbation types and five levels offer future robustness research. [R1,R2]

R2 warns that the actor source videos are not the real detector class: detector
controls are 1,000 FF++ target videos, excluded from its distribution and obtained
separately at c23. Thus testing on DeeperForensics after FF++ training risks shared
target content. R2 also provides variant metadata and reference splits; group
related variants and source identities in any future protocol. [R2]

Future official access requires reading its linked Terms of Use and submitting
the linked form; an educational email is preferred and a link follows approval.
Full PDF obligations remain unresolved. Its >=300 GB recommendation exceeds the
currently free D: storage even before processing. [R1,R2,R3]

## Strict split methodology (design only)

Random frame splitting can put nearly identical adjacent frames in train and
test. Shared backgrounds, identities, codec signatures and derivative content
then let a detector memorize familiar material. High apparent accuracy can
reflect that overlap rather than ability to detect unseen manipulations. Treat
videos and related sources, not individual frames, as the units of independence.

1. Build a versioned video inventory before extraction. Namespace IDs by dataset
   and version; preserve raw filename, compression, release partition, label,
   SHA-256, upstream URL/YouTube ID, and all known donor/target relationships.
2. Construct a lineage graph. Connect each derivative to its target original AND
   every known donor source; connect sequence segments from the same upstream
   recording, recompressions, crops, and confirmed duplicate/near-duplicate
   content. Every connected component is an indivisible split group. Identity
   grouping is preferred where reliable identity metadata exists; do not claim
   identity-disjoint evaluation from video IDs alone. Unknown lineage is quarantined.
3. Start with official split lists. FF++ pair IDs must both remain in the same
   group. Audit beyond pairs using conversion_dict and upstream recordings: 1,000
   sequences from 977 videos makes a sequence-only audit insufficient. Preserve
   original official assignments in metadata for traceability.
4. If a component crosses official splits, do not split the component to preserve
   published counts. Quarantine it from the experiment, recording the exclusion
   and resulting counts. Report this as an audited subset, not exact benchmark
   reproduction. If a new protocol is necessary, predeclare and version a grouped
   alternative separately. Keep independent evaluation contamination out of
   training, and do not reassign test data into training to solve a conflict.
5. Without suitable official splits, deterministically assign whole components
   with a recorded seed (proposed 42) and target ratio (proposed 70/15/15). Balance
   labels and methods at group level as feasible. Report achieved ratios, component
   sizes, exclusions and failures; never break a large group for nicer ratios.
6. Freeze split manifests and their SHA-256 checksums before frame extraction.
   Every derived frame, crop, augmentation and compression inherits that split.
   Exact byte hashes plus decoded-pixel hashes catch re-encoded duplicates;
   perceptual candidate matching followed by geometric/temporal comparison and
   human review addresses near duplicates. No perceptual threshold is claimed
   validated yet. Tune duplicate checks on development material, not test scores.
7. Re-audit extracted frames and cross-dataset sources before training. Required
   pass conditions: zero unresolved cross-split lineage edges, exact duplicates,
   confirmed near duplicates or source-video overlaps; complete provenance; valid
   labels; and verified applicable access. Any discovery after freezing blocks
   the run and creates a new audited manifest version with exclusion reasons.
8. Keep Celeb-DF wholly held out and DFDC development-only. Select checkpoint,
   thresholds, sampling, aggregation and any calibration on FF++ validation only;
   fit learned preprocessing/statistics on training only. Freeze the recipe before
   final within-dataset and independent testing. No final-test-driven retuning.

Illustrative examples (hypothetical; not dataset records):

- Originals A and B, fake A_B, reverse fake B_A, four manipulation versions and
  every extracted frame belong to component G1 and one split. A later B_C fake
  joins C to G1; grouping solely by the target A would miss this connection.
- Sequences S1 and S2 from one YouTube recording stay together even if the official
  list assigns them differently; quarantine the conflicting group in this policy.
- Frame 10 in train and re-encoded frame 11 in validation can be near duplicates
  despite different SHA-256 values. Review, group the sources and regenerate the
  split before training. Do not just remove one detected frame and keep its siblings.
- A DeeperForensics manipulation sharing an FF++ training target cannot be called
  independent evaluation. Exclude contaminated evaluation groups and disclose the
  reduced scope; a dataset name alone is not an independence guarantee.

Future metrics should include video-level ROC-AUC, PR-AUC, balanced accuracy,
confusion matrix and validation-chosen operating thresholds, with method and
quality breakdowns. Fix the aggregation recipe in advance. Bootstrap uncertainty
by source group, not frame; do not treat thousands of correlated frames as
thousands of independent test examples. These are planned evaluations only.

## Manifest contract

`docs/dataset_manifest_template.csv` contains exactly the requested 12 columns and
no example observations. One populated row represents one extracted frame (or one
video inventory entry with blank frame_index). Store populated manifests externally.

| Field | Definition and rule |
| --- | --- |
| dataset_name | Stable dataset identifier, e.g. faceforensics; never a personal filename |
| dataset_version | Exact release plus collection variant/compression identifier; do not mix v1/v2 |
| source_url | Public official dataset/release URL; never an expiring or signed access URL |
| access_date | ISO YYYY-MM-DD of actual authorized retrieval, not this research check date; blank until acquired |
| access_approved | true only with recorded approval/required terms acceptance; false if not approved; unknown if not established; only true is eligible |
| terms_reference | Stable reference into private access register identifying accepted terms snapshot/version and approval evidence |
| source_video_id | Namespaced ID of the exact video decoded, including variant/compression; all its frames share this ID |
| original_video_id | Namespaced target/base original; real video points to itself; unresolved fake base is quarantined; donor links belong in companion lineage metadata |
| label | real, fake or unknown, mapped from authoritative labels with mapping recorded; real means dataset annotation, not proven authenticity |
| split | train, validation, test, independent_test, development or quarantine; project assignment distinct from original release partition |
| frame_index | Zero-based decoded display-order frame number in source_video_id; blank only for video-level entries, never an invented timestamp |
| checksum | sha256: followed by 64 lowercase hexadecimal digits for the row's asset bytes: video for video rows, frame for frame rows |

The minimal columns cannot by themselves hold paths, multiple donors, or processing
history. A future companion asset index in processed\metadata must map
`(dataset_name,dataset_version,source_video_id,frame_index,checksum)` to a relative
asset path, asset type, parent video checksum, timestamp/time base, face/track ID,
crop coordinates, transform version and lineage group ID. A companion lineage
table records every donor, target, upstream recording and duplicate relation.
For multiple crops of one frame, checksum and face/track ID distinguish assets;
identical-content paths can be multiple locations for one checksum. Link each crop
to its parent full frame and each frame to its video inventory entry. Paths must
resolve under the external root. Freeze and hash all companion files with the
split manifest. Do not mistake a frame hash for its parent video's hash.

Traceability example in words: a frame row names fake A_B as source, A as original,
and decoded index 120; the asset index resolves that row to its frame path and
parent video hash, and lineage connects both A and donor B to the same group.
No actual records or synthetic training data are created in Phase 2.

## Future quality gates and reporting

Design only: implement these checks after a separately authorized pipeline phase.
All exclusions are explicit; never silently skip failures or alter official labels.

| Issue | Detection before training | Required report/action |
| --- | --- | --- |
| Missing files | Join inventory and asset index against actual paths | Missing/extra IDs and paths; block incomplete lineage groups |
| Corrupt images | Full decode, shape/channel checks, byte and pixel hashes | Asset/checksum/error; quarantine invalid assets and investigate parent |
| Unreadable videos | Probe stream metadata and decode selected frames; full scan for approved selected corpus | Codec, failure index, decode coverage; quarantine failures |
| Incorrect labels | Compare canonical labels with official metadata and method directories; review sampled cases | Conflict table with evidence; unknowns excluded, never relabel by model output |
| Class imbalance | Count videos, groups and frames per label/method/split | Counts/ratios; use training-only weighting/sampling, preserve test prevalence |
| Duplicate images | Compare SHA-256 and canonical decoded-pixel hashes globally | Duplicate clusters, splits and source IDs; cross-split conflicts block run |
| Near duplicates | Perceptual candidates plus temporal/geometric checks and reviewed samples | Candidate and confirmed clusters, thresholds/version, review outcomes |
| Source-video overlap | Intersect namespaced source/upstream IDs and group IDs across splits/datasets | Overlap matrix and offending groups; zero unresolved overlap required |
| Original/derivative leakage | Traverse donor/target and variant graph against assignments | Every crossing edge, group and split; quarantine/reversion split before training |
| Face detection failures | Record no face, multiple faces, low confidence, invalid boxes and track continuity | Rates by label/method/split; explicit abstention/selection policy, no success-only scoring |
| Inconsistent dimensions | Compare width/height/channels/orientation with declared crop contract | Histograms and outliers; preserve original and log any authorized resize |
| Video/frame inconsistency | Compare indices/timestamps/duration; spot-check paired original/fake timing and crops | Out-of-range, repeated or misaligned frames, variable-rate handling and sample discrepancies |
| Metadata inconsistencies | Schema, uniqueness, references, allowed values, checksums, approval and split validation | Row-level errors; block absent approvals, missing parents, stale hashes and ambiguous keys |

Each future report in reports should record run/time, dataset and manifest hashes,
tool/config versions, expected/observed counts, severity, affected IDs, decisions,
and before/after exclusion counts. Do not publish restricted imagery in reports.
Quality assessment must include failures by split and class so a detector's apparent
performance is not inflated by discarding difficult videos.

## Storage preparation and estimates

Observed D: availability on 2026-11-15: **100,767,629,312 free bytes**, approximately
100.77 decimal GB or 93.85 GiB, before directory creation. Existing used bytes:
4,088,918,016. Free space fluctuates; this is a snapshot, not reserved capacity.

Created the following 13 directories without overwriting files or deleting data:

```text
D:\MediaLensData\
  raw\
    faceforensics\
    dfdc\
    celeb_df\
  processed\
    frames\
    face_crops\
    metadata\
  manifests\
  models\
  reports\
  tmp\
```

Published estimates are in the comparison table. Current approved archive sizes
for DFDC and Celeb-DF are U; avoid quoting third-party sizes as verified. Full
DFDC remains explicitly excluded, regardless of future storage availability.

E: proposed initial budget: 10 GB FF++ c23, up to 5 GB for a small approved DFDC
package, 10 GB processed artifacts, 5 GB temporary staging, 1 GB metadata/reports
and later model artifacts, with at least 20 GiB free headroom. The DFDC allocation
is a ceiling, not a verified sample size. Celeb-DF storage is unallocated until
official package bytes are known. Do not fill the drive based on these estimates.

E: at 5,000 videos x 20 sampled frames = 100,000 frames, 224x224 RGB uncompressed
crops occupy 15,052,800,000 bytes (~15.05 GB), excluding headers. If actual encoded
crops average 20-60 KB, they would occupy ~2-6 GB; full-resolution frames can be
much larger. These assumptions must be measured on an approved pilot. Keeping
archives plus extracted videos plus frames multiplies storage. Plan capacity for
all simultaneous copies and filesystem overhead before any future acquisition.
No extraction or deletion is authorized by this budget.

## Remaining uncertainties and next-phase gates

- Actual access approval and download availability are unknown for every dataset.
  No forms, accounts or credentials were used in this session.
- Confirm current Kaggle sample inventory, labels, original references, size and
  accepted terms through an authenticated official page in a later access phase.
- Confirm Celeb-DF expanded v2 inventory and evaluation list on approval; clarify
  its broad copying/derived-data clauses before frame/crop processing.
- Read DeeperForensics full Terms of Use if revisited. Resolve donor/target and
  cross-dataset overlap before any future independent-test claim.
- Verify exact checksums and package sizes only after authorized acquisition;
  resolve DFDC preview's published count discrepancy against the release.
- Confirm whether intended downstream deployment, trained-model distribution and
  commercial use are permitted; software licenses do not settle dataset rights.
- Split purity, face availability, label quality, frame alignment, disk expansion
  and CPU runtime remain unmeasured. Official splits are proposals to audit, not
  evidence that the required leakage checks have already passed.

## Completion verification

On 2026-11-15, the existing backend command
`.venv\Scripts\python.exe -m pytest` passed all three tests outside the sandbox
(0.28 seconds, one existing Starlette/httpx deprecation warning). The first
sandboxed run stalled on a local TestClient request and was stopped. No dependency
change was made. `npm run build` passed using the existing frontend environment.
The build regenerated ignored production output as expected.

The external tree was inspected: 13 directories including the root, zero files.
The CSV has one header line and exactly 12 columns. Only the three requested
documentation files were authored. Modification timestamps on the 22 inspected
Phase 1 source/configuration/test files predate documentation creation; root
README, AGENTS.md and .gitignore also retain their earlier timestamps. No Git
repository exists here, so Git diff/status cannot establish an unchanged tree.
A pre-change hash snapshot was written to sandbox temporary storage but was not
recoverable afterward; byte-for-byte before/after verification is therefore not
claimed. No application or test source was edited by this task.

Phase 2 ends here. Dataset acquisition, extraction, preprocessing, dependency
installation and training require further instructions.
