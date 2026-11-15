# MediaLens AI — Cloud Deployment & Integration Guide

This guide details the deployment of the **MediaLens AI** backend to [Render](https://render.com) and connecting it to the production [Vercel](https://vercel.com) frontend at [https://medialens-ai.vercel.app](https://medialens-ai.vercel.app).

---

## 1. Architecture & Audit Summary

```
   ┌─────────────────────────────────────────────────────────────┐
   │                     Client Web Browser                      │
   └───────────────┬─────────────────────────────▲───────────────┘
                   │                             │ Direct CORS
                   │ HTTPS                       │ Requests (up to 50MB)
                   ▼                             │
    ┌──────────────────────────────┐             │
    │        Vercel (Edge)         │             │
    │  https://medialens-ai.       │             │
    │  vercel.app                  │             │
    │  (React 19 + Vite + TS)      │             │
    └──────────────────────────────┘             │
                                                 │
                                                 ▼
                                   ┌───────────────────────────┐
                                   │       Render Cloud        │
                                   │  (Web Service / Docker)   │
                                   │  FastAPI (app.main:app)   │
                                   │  OpenCV + PyTorch (CPU)   │
                                   └─────────────▲─────────────┘
                                                 │
                                                 │ HTTPS Download on Startup
                                                 │ (or Render Disk / Baked)
                                                 ▼
                                   ┌───────────────────────────┐
                                   │   Cloud Artifact Storage  │
                                   │ (S3 / R2 / GCS / Releases)│
                                   │   - best_model.pth        │
                                   │   - YuNet ONNX            │
                                   └───────────────────────────┘
```

### Component Audit

| Property | Value / Identification |
| :--- | :--- |
| **FastAPI Module** | `app.main` (file: `backend/app/main.py`) |
| **ASGI App Variable** | `app` (run string: `app.main:app`) |
| **Frontend Endpoints** | `GET /api/health`, `GET /api/v1/model/status`, `POST /api/v1/detect/image`, `POST /api/v1/detect/video` |
| **Frontend Base URL** | Configurable via `VITE_API_BASE_URL` (defaults to same-origin relative `/api/...`) |
| **Inference Engine** | CPU PyTorch 2.14.1 (MobileNetV3-Large) + OpenCV 5.0.0 (YuNet native DNN engine 2) |
| **CORS Configured** | `https://medialens-ai.vercel.app` + local development origins (`http://localhost:5173`, `5175`) |

---

## 2. Resource Constraints & Sizing Analysis

> [!WARNING]
> **Plan Selection Notice**: PyTorch and OpenCV video decoding have non-trivial memory and CPU requirements. Review the memory breakdown below before choosing an instance type.

### Memory & CPU Breakdown
- **Baseline Idle RSS**: ~300–350 MB (Python 3.12 + FastAPI + PyTorch CPU imports + OpenCV DNN engine + 26 MB MobileNetV3 weights).
- **Image Inference Peak**: ~380–420 MB (tensor transforms and single facial crop inference).
- **Video Inference Peak**: ~500–750 MB (sequential frame decoding of 1080p BGR frames, YuNet face detection, and 12-frame crop batch inference).
- **Execution Time**: Video analysis of a 120s video on 0.1 vCPU can take 45–80 seconds.

### Render Plan Comparison

| Plan | Specs | Suitability & Limitations |
| :--- | :--- | :--- |
| **Free Plan** | 512 MB RAM<br>0.1 shared vCPU | **Not recommended for video**: High risk of Linux OOM killer (`Exit Code 137`) during video decoding. Subject to 15-minute inactivity spin-down (cold start takes 50–90 seconds). Image inference works under strict single-request concurrency. |
| **Starter Plan** *(Recommended)* | 2 GB RAM<br>0.5 vCPU ($7/mo) | **Recommended**: Provides ample memory headroom for PyTorch + OpenCV video processing. Zero spin-down (no cold starts), eliminates request gateway timeouts. |

---

## 3. Model Checkpoint Hosting Requirements

The model checkpoint **must not be stored in Git** per project rules. The backend is configured to load the model from disk if present, or download it securely over HTTPS during startup or build.

### Required Artifacts

1. **MobileNetV3 Classifier Checkpoint**:
   - Filename: `best_model.pth` (~25.6 MB)
   - Local location: `D:\MediaLensData\models\medialens_mobilenetv3\best_model.pth`
   - Verified SHA-256: `5e19cc61d7b0c74b1f7979fb0682bc510dbc781c97d06f77945c50a25ab58622`
2. **YuNet Face Detector ONNX**:
   - Filename: `face_detection_yunet_2026may.onnx` (~224 KB)
   - Automatically retrieved from the official OpenCV Zoo release during deployment.
   - Verified SHA-256: `ebafce4e3c118d6554634be5c27ab333b4c047a9a8c3faf1d7cf93101c22f0f0`

### Uploading Your Checkpoint to Cloud Storage

Upload your local `best_model.pth` to any secure object storage provider that provides an HTTPS URL:

- **Option A (Cloudflare R2 / AWS S3)**:
  Upload `best_model.pth` to a private or public bucket. If using a pre-signed URL or public URL, copy the direct download link.
- **Option B (GitHub Release Asset)**:
  Create a private or public Release on your GitHub repository and attach `best_model.pth`. Use the direct asset download URL.
- **Option C (Hugging Face Model Hub)**:
  Upload to a Hugging Face model repository and use the `https://huggingface.co/.../resolve/main/best_model.pth` URL.

---

## 4. Step-by-Step Render Backend Deployment

### Method 1: Automated Blueprint via `render.yaml` (Recommended)

1. Ensure the updated repository is pushed to your Git provider (GitHub / GitLab).
2. Log into the [Render Dashboard](https://dashboard.render.com).
3. Click **New +** $\rightarrow$ **Blueprint**.
4. Connect your repository. Render will automatically detect the [render.yaml](file:///c:/Users/ayush/OneDrive/Desktop/medialens-ai/render.yaml) specification.
5. In the configuration screen, provide the secret environment variables:
   - `MEDIALENS_CHECKPOINT_URL`: The direct HTTPS download URL to your `best_model.pth`.
   - `MEDIALENS_AUTH_TOKEN`: *(Optional)* Bearer token if your storage requires authentication.
6. Click **Apply**. Render will build the container, install system libraries, pull YuNet, and start the service.

---

### Method 2: Manual Web Service Creation on Render

If you prefer configuring via the web UI without blueprints:

1. In Render Dashboard, click **New +** $\rightarrow$ **Web Service**.
2. Select your repository.
3. Configure the service settings:
   - **Name**: `medialens-ai-backend`
   - **Region**: Oregon (or your preferred region)
   - **Branch**: `main`
   - **Root Directory**: Leave blank (or `backend` if deploying native Python)
   - **Runtime**: **Docker** (Recommended: uses `backend/Dockerfile`)
     - *Dockerfile Path*: `./backend/Dockerfile`
     - *Docker Context*: `.`
   - **Instance Type**: **Starter** (2 GB RAM)
4. Set the **Health Check Path** to `/health` or `/api/health`.
5. Add the **Environment Variables**:

| Variable | Value | Description |
| :--- | :--- | :--- |
| `PORT` | `10000` | Port Render binds to (automatically assigned). |
| `OPENCV_FORCE_DNN_ENGINE` | `2` | Mandatory for YuNet native engine. |
| `MEDIALENS_ALLOWED_ORIGINS` | `https://medialens-ai.vercel.app` | Allows cross-origin requests from Vercel. |
| `MEDIALENS_MAX_CONCURRENT_INFERENCE` | `2` | Caps simultaneous heavy inference requests. |
| `MEDIALENS_CHECKPOINT_URL` | `<YOUR_CHECKPOINT_HTTPS_URL>` | Secure download URL for `best_model.pth`. |
| `MEDIALENS_AUTH_TOKEN` | *(Optional)* | Authorization bearer token for private storage. |
| `MEDIALENS_AUTO_DOWNLOAD_YUNET`| `1` | Automatically fetches verified YuNet ONNX. |

6. Click **Deploy Web Service**.
7. Once deployed, note your Render service URL: `https://medialens-ai-backend.onrender.com`.

---

## 5. Connecting the Vercel Frontend to Render

1. Open the [Vercel Dashboard](https://vercel.com/dashboard).
2. Select your `medialens-ai` project.
3. Navigate to **Settings** $\rightarrow$ **Environment Variables**.
4. Add a new variable:
   - **Key**: `VITE_API_BASE_URL`
   - **Value**: `https://medialens-ai-backend.onrender.com` (replace with your actual Render service URL, without trailing slash).
   - **Environments**: Select *Production*, *Preview*, and *Development*.
5. Save the variable.
6. Trigger a redeployment on Vercel:
   - Go to the **Deployments** tab.
   - Click the three dots `...` on the latest production deployment $\rightarrow$ **Redeploy**.
   *(Note: Vite bakes `import.meta.env` variables into the static bundle during `npm run build`, so a redeploy is required for the new variable to take effect).*

---

## 6. Verification & Smoke Testing

### 1. Test Backend Health & Readiness Directly
Run from any terminal:
```bash
# 1. Check service availability
curl -i https://medialens-ai-backend.onrender.com/health

# Expected response:
# HTTP/1.1 200 OK
# {"status":"ok","service":"medialens-ai"}

# 2. Check model readiness
curl -i https://medialens-ai-backend.onrender.com/api/v1/model/status

# Expected response:
# {"video_ready":true,"model_ready":true,"checkpoint_available":true,"model_version":"phase10-epoch3-...","threshold":0.5,"error":null}
```

### 2. Test CORS from Vercel Origin
```bash
curl -i -X OPTIONS https://medialens-ai-backend.onrender.com/api/v1/detect/image \
  -H "Origin: https://medialens-ai.vercel.app" \
  -H "Access-Control-Request-Method: POST" \
  -H "Access-Control-Request-Headers: Content-Type"

# Expected Header:
# access-control-allow-origin: https://medialens-ai.vercel.app
```

### 3. Verify Live Frontend
1. Open [https://medialens-ai.vercel.app](https://medialens-ai.vercel.app).
2. Look at the bottom status bar:
   - `Backend: connected` (green indicator).
   - `Model: ready · CPU` (green indicator).
3. Upload a facial image crop and click **Analyze Image**.
4. Switch to **Video Analysis**, upload an MP4/MOV clip, and verify video analysis results.

---

## 7. Troubleshooting & FAQ

### Backend shows `unavailable` on the frontend
- **Cold start delay**: If using Render's Free tier, the service spins down after 15 minutes of inactivity. The initial wake-up can take 45–75 seconds. The frontend timeout has been increased to 35 seconds; click `Check connection ↻` after 45 seconds.
- **Incorrect VITE_API_BASE_URL**: Ensure there is no trailing slash (e.g., `https://my-app.onrender.com`, not `https://my-app.onrender.com/`).
- **Missing Vercel redeploy**: Ensure you redeployed the Vercel project after adding the environment variable.

### `Model unavailable. Check backend configuration` (HTTP 503)
- Check Render application logs for model startup messages.
- Ensure `MEDIALENS_CHECKPOINT_URL` is accessible from the Render container and points directly to the file bytes.
- Check that the file checksum matches `5e19cc61d7b0c74b1f7979fb0682bc510dbc781c97d06f77945c50a25ab58622`.

### Container crashed with `Exit Code 137`
- Exit Code 137 indicates the Linux Out-Of-Memory (OOM) killer terminated the process.
- Upgrade the Render instance from Free (512 MB) to Starter (2 GB RAM).

### CORS Error in Browser Console (`No 'Access-Control-Allow-Origin' header is present`)
- Verify that your Render service environment variable `MEDIALENS_ALLOWED_ORIGINS` includes `https://medialens-ai.vercel.app`.

