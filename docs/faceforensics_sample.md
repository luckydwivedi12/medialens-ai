# FaceForensics++ sample manifest — Phase 4B

Verified on **2026-11-15**. This is a **pipeline smoke-test sample**, not a
train/validation/test benchmark. No additional media was downloaded, no frame
files were extracted, and no models were trained.

## Actual sample and source evidence

The downloaded sample contains 20 original videos and 20 manipulated Deepfakes
videos, all at c23: 40 MP4 files totaling **78,201,925 bytes**. These figures were
checked against both the supplied inventory and a new read-only audit. No other
manipulation method or compression level is represented.

```text
D:\MediaLensData\raw\faceforensics\
  original_sequences\youtube\c23\videos\        20 originals
  manipulated_sequences\Deepfakes\c23\videos\   20 manipulated videos
```

The [official dataset documentation](https://github.com/ondyari/FaceForensics/blob/master/dataset/README.md),
checked on 2026-11-15, defines manipulated filenames as
`<target_sequence>_<source_sequence>` and original sequence IDs as integers from
0 to 999. Preserve leading zeroes in this sample. For `033_097.mp4`, target/base
033 receives the face from donor/source 097. `097_033.mp4` reverses the direction.
The filename convention establishes these sequence relationships; it does not
establish broader identity or upstream-recording independence.

Original inventory used:
`D:\MediaLensData\reports\dataset_inventory_20261115T082932429305Z_78a489446d714fe3a39a40453bb88c44.csv`.
Its accompanying audit ended at 08:29:40 UTC with 40 valid files, zero invalid
files, no exact duplicate groups and one missing-manifest warning.

## Access provenance

The owner explicitly attested in this chat that email-based authorization/download
instructions were received on **2026-11-15**:

- Sender: noreplyfaceforensics@gmail.com.
- Subject: FaceForensics and FaceForensics++ Dataset.
- Download script: [author-supplied faceforensics_download_v4.py](http://kaldir.vc.in.tum.de/faceforensics_download_v4.py).
- Sample downloaded that day using the EU2 server.
- Terms displayed by the script were accepted before continuation.
- Terms reference: [FaceForensics Terms of Use](http://kaldir.vc.in.tum.de/faceforensics/webpage/FaceForensics_TOS.pdf).

The manifest records `access_date=2026-11-15` and `access_approved=true` on the
basis of that user attestation. `access_evidence` explicitly identifies the
email-based authorization and its provenance. The email was not independently
opened or authenticated. There is no separate approval certificate or reference
number, and none was invented. The links above are references supplied by the
owner; this task did not execute or download the script or reaccept terms.

This is not a grant of broader commercial or redistribution rights. The existing
Phase 2 dataset usage restrictions continue to apply to media and derived data.

## Deterministic leakage groups

All records have `split=unassigned` and `sample_purpose=pipeline_smoke_test`.
No train, validation, or test allocation was made. Group IDs use the two
zero-padded original IDs sorted lexicographically: `ffpp-c23-pair-A-B`.

| leakage_group_id | Original IDs | Manipulated video IDs |
| --- | --- | --- |
| ffpp-c23-pair-033-097 | 033, 097 | 033_097, 097_033 |
| ffpp-c23-pair-183-253 | 183, 253 | 183_253, 253_183 |
| ffpp-c23-pair-210-241 | 210, 241 | 210_241, 241_210 |
| ffpp-c23-pair-252-266 | 252, 266 | 252_266, 266_252 |
| ffpp-c23-pair-339-392 | 339, 392 | 339_392, 392_339 |
| ffpp-c23-pair-469-481 | 469, 481 | 469_481, 481_469 |
| ffpp-c23-pair-585-599 | 585, 599 | 585_599, 599_585 |
| ffpp-c23-pair-672-720 | 672, 720 | 672_720, 720_672 |
| ffpp-c23-pair-828-830 | 828, 830 | 828_830, 830_828 |
| ffpp-c23-pair-866-878 | 866, 878 | 866_878, 878_866 |

Each group has exactly two real and two fake records. The ten groups are disjoint
under the verified filename relationships, not proven independent identities or
upstream recordings. A future split must keep each group together and perform
the broader Phase 2 overlap audit before claiming split independence.

## Manifest and schema compatibility

Manifest: `D:\MediaLensData\manifests\faceforensics_sample_manifest.csv`.

SHA-256 of the completed CSV:
`d7f58e94a67e04482b4e8adaaa09630a76dd42679d989fc80ade4ce28445690f`.

The existing checker was inspected before creation. It requires all 12 columns
from [the unchanged template](dataset_manifest_template.csv), accepts unique
additional columns, and matches inventory assets using SHA-256. This manifest
retains all 12 required columns and adds eight optional fields. It has **20
columns**, 40 data records and a header.

| Field | Meaning in this sample |
| --- | --- |
| dataset_name | faceforensics-sample; matches the audit command |
| dataset_version | FaceForensics++-Deepfakes-c23; a descriptive collection/variant identifier, not an invented official release number |
| source_url | Public official dataset README URL |
| access_date | User-confirmed retrieval date, 2026-11-15 |
| access_approved | true, based on user-attested email authorization and terms acceptance |
| terms_reference | Owner-supplied Terms of Use URL above |
| source_video_id | Exact video being inventoried: 033 for the original, 033_097 for that manipulated asset; NOT the face donor |
| original_video_id | Self for real videos; target/base original ID for fake videos |
| label | real or fake according to dataset folder and naming, not an authenticity prediction |
| split | unassigned for every record |
| frame_index | Blank: these are video-level records, not extracted frames |
| checksum | sha256: plus 64 lowercase hexadecimal digits for the video bytes |
| video_id | Filename stem preserving zeroes, unique within this collection |
| relative_path | Forward-slash path relative to the external dataset root |
| target_video_id | Target/base original ID; self for real records |
| donor_video_id | Face donor original ID for fake records; blank for real records |
| manipulation_method | Deepfakes for fake records, none for real records |
| leakage_group_id | Deterministic sorted-original-pair group above |
| sample_purpose | pipeline_smoke_test |
| access_evidence | User-attested email sender, subject, date, download procedure and terms acceptance; explicitly no certificate/reference number |

All references resolve within this dataset_name/dataset_version namespace.
The original_video_id remains the actual target; it was not replaced with an
artificial group representative to force grouping.

The compatibility extension adds `unassigned` as a valid split and emits one
aggregate `unassigned_split` warning. Optional donor_video_id edges now contribute
to the existing original/derivative component audit, and optional
leakage_group_id values are checked for cross-split assignments. Old manifests
without either optional field retain their previous behavior. Other optional
fields are descriptive; the sample builder performs their integrity checks.
These additions supersede the Phase 3 documentation's earlier split vocabulary
and statement that donor relationships are not interpreted. Historical Phase 2
and Phase 3 documents were preserved without rewriting them.

## Integrity and audit results

All ten requested integrity checks passed:

1. Exactly 40 unique video records.
2. Exactly 20 real records.
3. Exactly 20 fake records.
4. All 40 relative paths exist and match the disk MP4 inventory.
5. All video hashes match both the original inventory and freshly hashed bytes.
6. Each fake's target and donor match its filename.
7. Every referenced original is present and labeled real.
8. Exactly ten deterministic groups exist.
9. Each group contains exactly two real and two fake records.
10. Both reciprocal manipulations and their two originals share the correct group.

The existing audit CLI was run from the repository root:

```powershell
.\backend\.venv\Scripts\python.exe backend/scripts/audit_dataset.py --dataset-root "D:\MediaLensData\raw\faceforensics" --dataset-name "faceforensics-sample" --manifest "D:\MediaLensData\manifests\faceforensics_sample_manifest.csv"
```

New, exclusively created reports:

- JSON: `D:\MediaLensData\reports\dataset_audit_20261115T084010728180Z_7bd9b882e54d46feb8704409074eca25.json`.
- CSV: `D:\MediaLensData\reports\dataset_inventory_20261115T084002294908Z_d2d3bf0c8fbc4866b8b88e31124f4e51.csv`.

Audit completed at **08:40:10 UTC / 14:10:10 IST** on 2026-11-15.
Result: **warning**, 40 valid videos, zero invalid videos, zero traversal errors,
zero exact duplicate groups, zero missing-metadata issues, zero reported leakage
violations, zero errors and **one warning**. That warning is `unassigned_split`,
covering all 40 rows. It is deliberate: split leakage has not been meaningfully
tested between train/validation/test sets because those sets do not yet exist.
The prior missing-manifest warning is resolved, not suppressed.

## Implementation and preservation checks

Added `backend/scripts/create_faceforensics_sample_manifest.py` for reproducible,
read-only construction and integrity validation of this exact 40-video reciprocal
sample shape. It rejects unexpected counts, names, missing originals/reciprocals,
overlapping pair membership and inventory/hash discrepancies; it never repairs
them. Output is exclusive-create and must be outside the raw dataset root.
Running it again against the existing output refuses to overwrite the manifest.

Added five builder tests and three checker regression tests. **48 backend tests
passed**, including existing API/audit tests; only the pre-existing Starlette/httpx
deprecation warning remains. The frontend production build passed. No dependency
changes were needed. The only modified pre-existing code files were
`leakage_check.py` and its tests; 38 other baseline files remained byte-identical.
All four pre-existing reports remained byte-identical. All 40 media hashes remain
identical to the supplied inventory after the audit. No Git repository is
initialized, and no commits, pushes, or GitHub changes were made.

## Limits and next-phase boundary

Forty correlated videos represent only ten filename-derived leakage groups. Splitting
these few groups would yield tiny, unstable estimates and cannot support meaningful
model training, model selection, threshold tuning or a generalization benchmark.
The sample covers one manipulation method and one compression setting; it is not
representative of the full FF++ dataset or real-world manipulation prevalence.

OpenCV checked three selected frames per video, not every frame. No near-duplicate,
identity, original-YouTube-recording, donor-model provenance or cross-dataset audit
was performed. There is no inference or face detector in this phase. Passing the
current checks does not establish that leakage is impossible, that every frame is
uncorrupted, or that any video is definitively authentic.

Phase 4B ends here. Further acquisition, frame extraction, split assignment,
preprocessing and training require further instructions.
