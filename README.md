# MediaLens AI

A local experimental deepfake assessment application. Image analysis applies the
trained CPU MobileNetV3-Large to an uploaded facial crop. Phase 12 video analysis
uses YuNet faces from 12 uniformly sampled frames and aggregates the existing
classifier's scores. Video performance has not been validated; predictions and
aggregate scores are not proof of authenticity.

See [application setup](docs/application_setup.md) for startup commands and image
API details, and [Phase 12 video detection](docs/video_detection.md) for video
formats, limits, sampling, scores, verification and limitations. Earlier phases
are preserved. No training or new ML dependencies are required in this workspace.

## Prerequisites

- Windows 11
- Python 3.12.10 (64-bit)
- Node.js 24.16.0 and npm 11.13.0

## Backend setup

Run in PowerShell from the project root:

```powershell
python -m venv backend/.venv
cd backend
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
```

For normal local startup, run `.\scripts\start_local.ps1` from the project root and press Ctrl+C in that console to stop both services.

For manual startup, run the API from `backend` (no environment activation required):

```powershell
.\.venv\Scripts\python.exe -m uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

`GET http://127.0.0.1:8000/api/health` returns:

```json
{"status":"ok","service":"medialens-ai"}
```

Interactive API documentation is at `http://127.0.0.1:8000/docs`.
CORS permits the local Vite origins on port 5173.

## Frontend setup

In a second PowerShell terminal, from the project root:

```powershell
cd frontend
npm ci
npm run dev
```

Open `http://localhost:5175`. Vite proxies `/api` requests to
`http://127.0.0.1:8000`. Start both servers to see a connected status.
The interface has Image Analysis, Video Analysis and About modes, upload previews,
experimental assessments and backend/model readiness indicators. Video accepts
MP4/MOV/AVI/MKV up to 50 MB and 120 seconds using temporary storage only. Image
analysis expects a facial crop; video locates and crops faces automatically.
Neither mode analyzes audio. See the setup and video guides above.

`npm run preview` previews the production assets only. Production API routing
must be configured separately; the `/api` proxy is a development feature.

## Verification

From `backend`:

```powershell
.\.venv\Scripts\python.exe -c "from app.main import app; print(app.title)"
.\.venv\Scripts\python.exe -m pytest
.\.venv\Scripts\python.exe -m pip check
```

From `frontend`:

```powershell
npm test
npm run build
```

The installed Starlette version emits a deprecation warning for its `httpx`
TestClient integration. Tests pass with the requested `httpx` dependency; no
additional HTTP client is installed in Phase 1.

## Structure and responsibilities

```text
backend/
  app/
    __init__.py
    main.py
    api/
      __init__.py
      routes/
        __init__.py
        health.py
    core/__init__.py
    schemas/
      __init__.py
      health.py
    services/__init__.py
  tests/test_health.py
  requirements.txt
  requirements-dev.txt
frontend/
  src/
    App.tsx
    main.tsx
    styles.css
  index.html
  package.json
  package-lock.json
  tsconfig.json
  tsconfig.app.json
  tsconfig.node.json
  vite.config.ts
AGENTS.md
README.md
.gitignore
```

API routes, schemas, and services have separate responsibilities. Future media
preprocessing and inference belong in separate service modules, outside route
handlers and UI code. CPU inference is the initial target for this machine.
PyTorch, TensorFlow, OpenCV, datasets, and model weights are not part of Phase 1.

Never present future AI predictions as definitive proof of authenticity.
Virtual environments, dependencies, builds, datasets, uploads, and model weights
are excluded by `.gitignore`. The workspace is inside OneDrive; synchronization
of generated files may require local OneDrive settings later.

No Git repository initialization, GitHub connection, commits, or pushes are
required to run this project.
