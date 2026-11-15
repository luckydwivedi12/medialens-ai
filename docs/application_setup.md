# MediaLens AI local application

## Run the existing environment

From the project root in PowerShell, run:

```powershell
.\scripts\start_local.ps1
```

Keep that console open. Press **Ctrl+C in the launcher console** to stop both services. The launcher checks ports before starting, uses the project virtual environment, and waits for both models through the frontend proxy. If either port is occupied, inspect its owner before restarting the launcher.

Open <http://localhost:5175> for the interface and <http://127.0.0.1:8000/docs> for the API. Release 0.12.0 serves health, model status, image detection and video detection on port 8000. Vite forwards same-origin /api requests there. The launcher binds both services to loopback.

For a fresh environment, install the existing pinned `requirements-dev.txt`
and `requirements-ml.txt` inside `backend/.venv`, and run `npm ci` in `frontend`.
The application does not download weights at startup. Keep the existing CPU
PyTorch/TorchVision versions; this phase installs no new dependencies.

## Model configuration and lifecycle

Default checkpoint:
`D:\MediaLensData\models\medialens_mobilenetv3\best_model.pth`.
To select a different trusted, compatible checkpoint before starting FastAPI:

```powershell
$env:MEDIALENS_CHECKPOINT = 'D:\MediaLensData\models\medialens_mobilenetv3\best_model.pth'
```

The checkpoint must have its original sibling `model_metadata.json` and
`preprocessing_config.json`. Startup verifies its SHA-256 against metadata,
format, exact architecture, class mapping, preprocessing and strict state-dict
compatibility. It constructs the existing MobileNetV3-Large with its trained
two-logit head, no pretrained download, and loads on CPU with `weights_only=True`.
It retains one evaluated model per backend process, uses six CPU threads and
`torch.inference_mode()`, and serializes forwards with a lock. Restart after
changing configuration; a failed load does not silently retry or invent scores.
Use one backend worker on this hardware.

Saved checkpoint and metadata confirm **real=0, fake=1**. Exact shared Phase 10
inference transform: RGB, direct 224×224 resize with antialiased bilinear
interpolation, float [0,1], normalize with mean [0.485, 0.456, 0.406] and std
[0.229, 0.224, 0.225]. No random augmentation or ImageNet center crop is used.
Uploads have EXIF orientation applied before this transform. The original
training crops are already orientation-free PNGs.

The saved model metadata specifies **decision_threshold=0.5**; Phase 10's
original inference rule also uses `p_fake >= 0.5`. This is the documented fixed
development rule, not a newly tuned or calibrated threshold. The loader falls
back to that same documented rule if a compatible metadata file omits the field.
Softmax uses the class dimension of two finite logits; named probabilities use
the validated checkpoint mapping. Version identifies epoch and checkpoint hash.

## API

`GET /api/health` is independent backend liveness. Existing response preserved:

```json
{"status":"ok","service":"medialens-ai"}
```

`GET /api/v1/model/status` reports startup readiness, architecture, CPU device,
version, checkpoint availability at startup, threshold and a sanitized error.
Example from the real loaded checkpoint:

```json
{
  "model_ready": true,
  "architecture": "MobileNetV3-Large",
  "device": "cpu",
  "model_version": "phase10-epoch3-5e19cc61d7b0",
  "checkpoint_available": true,
  "threshold": 0.5,
  "error": null
}
```

On failure, `model_ready` is false, version and threshold are null, and `error`
explains that configuration must be checked and the backend restarted. A present
but invalid checkpoint may have `checkpoint_available=true` and readiness false.
The UI distinguishes backend connectivity from model readiness and disables
analysis while the model is unavailable.

`POST /api/v1/detect/image`: multipart form with field **file**, containing one
JPEG/JPG (`image/jpeg`), PNG (`image/png`) or still WEBP (`image/webp`). Example:

```powershell
curl.exe -X POST http://127.0.0.1:8000/api/v1/detect/image `
  -F "file=@C:\path\to\your\facial_crop.png;type=image/png"
```

Filenames are never used as filesystem paths. MIME must match the actual decoded
image format. Limits: 10 MiB per image (shown as 10 MB in UI), 20 million decoded
pixels, no animation. Multipart request bodies are bounded at 10 MiB plus 64 KiB
overhead, including absent/dishonest Content-Length and chunked uploads. Pillow
verifies structure, fully decodes, applies EXIF transpose and converts to RGB.
Processing uses memory; Starlette may spool multipart parts to OS temporary files,
which are closed and removed after processing. No upload directory, image archive
or permanent user-image retention is created. No weights or paths are returned.
CORS allows only `http://localhost:5173` and `http://127.0.0.1:5173`, GET/POST,
Content-Type, without credentials.

Actual response for validation sample `004_982:f92:d0`:

```json
{
  "success": true,
  "predicted_label": "fake",
  "probabilities": {"real": 0.23164042830467224, "fake": 0.7683594822883606},
  "threshold": 0.5,
  "model_name": "MobileNetV3-Large",
  "model_version": "phase10-epoch3-5e19cc61d7b0",
  "device": "cpu",
  "warning": "Experimental research model. Prediction is not proof of authenticity. Scores are not calibrated probabilities of real-world authenticity."
}
```

Errors consistently return `success=false` and `error: {code, message}`. Statuses:
400 invalid/empty/corrupt/mismatched/animated/over-pixel-limit image; 413 oversized
image/request; 415 unsupported MIME; 422 missing multipart file; 503 model
unavailable; 500 inference failure. Exception details stay in backend logs.

## Verification and real-checkpoint smoke

```powershell
cd C:\Users\ayush\OneDrive\Desktop\medialens-ai\backend
.\.venv\Scripts\python.exe -m pytest
.\.venv\Scripts\python.exe scripts\smoke_application.py
cd ..\frontend
npm test
npm run build
```

The smoke script requires both services running. It reads the checksum-verified
split manifest as metadata, selects a paired **validation** crop before accessing
any image path, verifies that crop's checksum, and sends it to both live API and
Vite proxy. It independently reproduces the Phase 10 dataset transform and model
forward pass, checks named API probabilities within 1e-6, verifies repeatability
and checks the checkpoint SHA-256 before/after. It writes no image or report file.
It never resolves, stats, hashes or opens test-image paths.

Results:

- Backend: **137 passed**, including 20 new API/inference/upload tests. The existing
  Starlette/httpx deprecation warning remains; no dependency added to suppress it.
- Frontend: **6 tests passed**; production TypeScript/Vite build passed. Node's built-in tests cover
  file validation, prediction schema, multipart API request, structured errors,
  malformed responses and abort propagation, without adding dependencies.
- Live backend and frontend proxy returned identical real checkpoint scores above;
  independently computed scores match, repeated responses match, checkpoint unchanged.
- Checkpoint SHA-256:
  `5e19cc61d7b0c74b1f7979fb0682bc510dbc781c97d06f77945c50a25ab58622`.
- Zero locked-test images opened. No retraining, dataset changes, checkpoint
  changes, new ML dependencies, commits or GitHub push.

Windows sandbox restrictions blocked temporary directories and local sockets in
the first test attempts. The passing full run and live smoke were executed with
approved execution outside that sandbox.

Browser UI verification remains incomplete: the in-app browser could not connect
to localhost, and the local Computer Use runtime then stopped because it could not
determine the browser URL confidently enough to enforce policy. No claim is made
that live drag/drop, preview/remove/replace, loading/error/retry rendering, final
prediction rendering or responsive visual layout were observed in a browser.
These behaviors are implemented; manually check them in the UI before relying on
the app. API tests and proxy smoke do not replace those browser checks.

## Scope and limitations

This is an image classifier trained on **facial crops**, not a full-image face
detection pipeline. Upload a single facial crop for the closest input match. It
does not auto-detect faces, analyze video/audio or establish authenticity. The
small FaceForensics++ matched subset covers Deepfakes/c23 only; labels remain
provisional, identity independence is unverified, and other methods/sources or
full images may be outside its training distribution. No performance metric is
added or inferred from the smoke prediction. Consult `model_training.md` for
existing development evaluation and limitations.

This local research server has no authentication, public deployment hardening,
global rate limit or multi-user job queue. Keep it bound to loopback. It retains
only the selected preview in browser memory until removal/replacement/page close.
The backend does not retain images permanently.

## Phase 11 file inventory

Created:

- `backend/app/api/routes/detection.py` — prediction and model status endpoints.
- `backend/app/schemas/detection.py` — public typed responses and research warning.
- `backend/app/services/images.py` — upload/image validation and EXIF-aware decode.
- `backend/tests/test_detection.py` — 20 bounded unit/API cases.
- `backend/scripts/smoke_application.py` — read-only real-checkpoint live smoke.
- `frontend/src/api.ts` — multipart client and prediction response validation.
- `frontend/src/upload.ts` — shared frontend file validation.
- `frontend/tests/api.test.ts` — dependency-free frontend contract tests.
- `docs/application_setup.md` — this guide and verification record.

Modified:

- `backend/app/main.py` — startup lifecycle, bounded body middleware, CORS and errors.
- `backend/app/ml/inference.py` — singleton image inference service; Phase 10 helpers preserved.
- `frontend/src/App.tsx` — full upload, preview, analysis, results, status and About UI.
- `frontend/src/styles.css` — responsive warm off-white/charcoal/sage styling.
- `frontend/index.html` — current image-analysis description.
- `frontend/package.json` — built-in Node test command.
- `README.md` — current application summary and setup guide link.

Build output under `frontend/dist` is regenerated and ignored. Model weights,
datasets and uploads remain outside source control. Phase 11 ends here; no later
training, calibration, locked-test evaluation or deployment phase was started.


## Phase 12 extension

Video analysis is now available alongside the preserved image workflow. See
[video_detection.md](video_detection.md) for `POST /api/v1/detect/video`, YuNet
configuration, supported formats, limits and actual verification. Model status
adds `video_ready`; restart the backend after updating. The earlier Phase 11
results and limitations above describe that phase as it was completed.
