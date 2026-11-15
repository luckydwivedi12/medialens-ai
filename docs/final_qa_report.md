# MediaLens AI — final QA and security audit

## Phase 14.1 release consolidation

The local release blocker is resolved. Start the consolidated services with `scripts/start_local.ps1` from the project root and stop both with Ctrl+C in its console. See [application setup](application_setup.md).

The previous port 8000 listener (PID 23136, release 0.11.0) and alternate port 8001 listener (PID 35216, release 0.12.0) ran the same Uvicorn command from different launch times. Both used the project backend virtual-environment launcher and its Python 3.12 base interpreter. The stale process retained its original routes in memory. The previous Vite process (PID 6704) used the alternate proxy target. After verifying project ownership, dedicated-console Ctrl+C stopped both Uvicorn services and the project Vite listener was replaced.

At verification, backend PID 3888 listened on 127.0.0.1:8000 and frontend PID 16680 listened on IPv6 loopback port 5175. Port 8001 had no listener. A second launcher invocation refused occupied port 8000. Direct OpenAPI reported application 0.12.0 and specification 3.1.0 with GET health, GET model status and both detection POST routes. Health returned ok and both classifier and video readiness were true.

Fresh regression passed all 155 backend tests and 13 frontend tests, the forced TypeScript check and the production build. Real validation image and video smoke tests passed directly against port 8000 and through port 5175. An independent checkpoint forward reproduced image fake score 0.7683594822883606. For video, the smoke separately decoded the reported frames, cropped the returned boxes and compared both class scores on every analyzed face against CPU checkpoint inference within 1e-6. Its aggregate and predicted label agreed with the API. The video mean fake score was 0.734908347328504 across 12 analyzed frames. Checkpoint integrity and temporary video cleanup checks passed. This verification covers supervised loopback research use. The Phase 14 public deployment findings below still apply.

## Historical Phase 14 audit before consolidation

**Decision: suitable for supervised local research through the verified frontend. Do not sign off the advertised port-8000 video API or a public deployment.** The underlying source passes regression. The two running backends do not serve the same release.

## Findings and disposition

| Priority | Finding and evidence | Action |
|---|---|---|
| Release blocker for direct API clients | Port 8000 serves OpenAPI version 0.11.0, with health, model status and image routes only. POST video returned 404. Port 8001 serves 0.12.0; the same request without a file returned the expected 422. The frontend on 5175 successfully analyzed video via its proxy. | At a coordinated restart, serve the current backend on the intended port and align the proxy. Verify video_ready, the route list and both smoke tests before retiring the alternate service. This audit did not disrupt the running servers. |
| High for public exposure | File and frame limits bound individual jobs, but concurrent uploads can spool to disk and queue behind a process-local inference lock. Native decode and inference have no hard timeout. There is no global admission limit. | Keep loopback-only. Before shared deployment, bound concurrency, temporary storage, worker memory and execution time; test overload, cancellation and worker recovery. |
| Data hygiene | Eleven Phase 13 screenshots include UI evidence, some showing validation imagery. Existing ignores did not cover the evidence directory or common video extensions everywhere. | Added exclusions for that directory, video extensions, reports, masks and face crops. Preserved all files. New browser evidence and raw smoke JSON went under ignored test temporary storage. |
| Local browser threat | CORS is a browser response-reading policy. It does not authenticate clients or prevent a simple cross-origin form submission from consuming CPU. | Do not rely on CORS as an access-control or resource-abuse defense. Review Origin/Host and request admission before wider use. |

## Fresh regression evidence

| Check | Outcome |
|---|---|
| Entire backend suite | 155 passed; one existing Starlette/httpx deprecation warning |
| Entire frontend suite | 13 passed |
| TypeScript and production assets | Forced project type check ran; production TypeScript/Vite build passed |
| Python environment | pip check: no broken requirements |
| Image end-to-end | Port 8000 and frontend proxy matched independent Phase 10 forward pass and repeated predictions |
| Video end-to-end | Port 8001 and proxy matched per-frame results. Direct port 8000 failed route discovery and POST (404). |
| Headless Edge | Browser audit completed against 5175, without uncaught JavaScript exceptions or console.error calls during the run |

Image tests used generated JPEG, PNG and still WEBP uploads; invalid bytes, truncated input, empty files, wrong or mismatched MIME, oversized/chunked uploads, animation, excessive pixels, EXIF/RGB handling, missing checkpoint, unavailable model and sanitized inference exceptions were exercised. The real smoke used only checksum-verified validation crop 004_982:f92:d0. Output: real 0.23164042830467224, fake 0.7683594822883606. The independent forward pass used the saved mapping real=0/fake=1 and deterministic RGB, antialiased bilinear 224-square resize and ImageNet normalization. The fixed fake threshold is inclusive at 0.5.

Video tests exercised generated AVI, MOV, MKV and MP4 uploads; corruption, mismatched container, unsupported format, empty and oversized uploads, duration rejection, distinct sampling, invalid crops, partial decode failures, missing model, no faces, two analyzed frames, unequal face counts and temporary cleanup after both success and exceptions. Controlled detector/classifier fixtures establish edge-case behavior; they do not measure model performance. The real smoke used validated development video 004_982, actual YuNet crops and the shared checkpoint: 12 sampled and analyzed frames, 12 faces, zero no-face or failed frames, mean fake score 0.734908347328504. All 12 frame means met the threshold. Independent arithmetic over returned face scores matched the response. Temporary video directories returned to baseline and source/checkpoint hashes were unchanged. Neither smoke opened locked-test media.

## Security checks and limits

- Upload routes compare claimed type to decoded image format or video container signature and require actual decoding. Limits are 10 MiB / 20 megapixels for images and 50 MiB / 120 seconds for videos. Video validation also bounds dimensions and FPS. Multipart overhead is separately bounded, including chunked requests. Container signatures and successful decoding do not certify arbitrary hostile media as safe.
- Uploaded filenames are never used as storage paths. The video route uses a server-created temporary directory and a fixed basename with an allowlisted suffix. Traversal-style filenames, cleanup, and absence of private paths in representative error/success responses were regression-tested. Managed cleanup does not cover abrupt process termination or guarantee secure erasure.
- Live CORS preflight: localhost:5173 received 200 and its allowed origin; an untrusted origin received 400 and no allow-origin header. The 5175 frontend uses a same-origin proxy. Credentialed CORS is disabled.
- Route inspection and live OpenAPI inventories showed no checkpoint-serving route. The loader checks metadata/checksum, strict architecture and class mapping, CPU placement and weights-only deserialization. Metadata and checkpoint are local trust inputs: a hash alongside a writable model does not authenticate its publisher.
- Inspected application logging: routes log operational exceptions, not uploaded bytes. Tracebacks can contain server paths in operator logs. Tests check sanitized client errors; this is not an exhaustive inspection of OS, proxy, native decoder or third-party logging.
- Reviewed Python requirement pins, CPU wheel index, frontend manifest and lockfile. Dependency consistency and successful build are not vulnerability scans. No current advisory clearance, native-code fuzzing, penetration-test certification or production load test is claimed.
- No Git repository was found at or above the workspace. There is no local index/history to attest to. A filename/size review excluding installed dependencies, builds and known temporary directories found the prior screenshots, no candidate model/video artifacts or other files above 5 MiB in the scanned source tree. A filename-only-output scan for common private-key, AWS, GitHub and OpenAI credential signatures returned no matches. Pattern checks cannot exclude arbitrary secrets. Before initializing or publishing a repository, review staged contents, especially image formats and generated documents outside the known directories. No commit or push occurred.

## Frontend and accessibility

The browser selected and replaced a validation image, dispatched a drop, displayed the preview and real result, and surfaced an invalid replacement. It exercised video selection, preview, visible processing, result and frame disclosure, navigation and About. The initial screen and expanded video results fit 1440, 1366, 768 and 390 CSS pixels without document-level horizontal overflow. Keyboard Tab reached visible focus. Component tests also exercised pending/error transitions and insufficient-result rendering. The UI states that aggregate scores are uncalibrated; these smoke predictions are not accuracy metrics or proof of authenticity. No training or model updates took place.

Limitations: one desktop browser with emulated widths; no full keyboard-only journey, physical mobile, assistive-technology, exhaustive codec or WCAG audit. Offline-backend recovery and every failure state were not independently reproduced in a browser. Prior reports remain intact; Phase 14 changes were limited to hygiene, a configurable browser-audit evidence destination, and this report.

## Reproduction

From backend run the existing virtual environment's Python with -m pytest -q and -m pip check. Run scripts/smoke_application.py with --backend http://127.0.0.1:8000 --frontend http://localhost:5175; use port 8001 for scripts/smoke_video.py until the release mismatch is resolved. Run scripts/audit_ui.py with MEDIALENS_AUDIT_OUTPUT set to a new private evidence directory under an existing parent. From frontend run npm test, the local tsc -b --force, and npm run build. Keep smoke selection restricted to the verified development manifest.
