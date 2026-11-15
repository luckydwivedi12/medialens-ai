# Phase 13 — UI refinement report

The existing image and video screens continue to use real backend inference.

| Area | Changes |
|---|---|
| Branding and navigation | Changed the hero to “Review facial media”, used “Media forensics” in the hero and footer, and made the brand link return to Image Analysis. About now explicitly names model scope and limitations. |
| Workflow copy | Replaced decorative empty-state headlines with awaiting-analysis headings. Image loading names the facial crop being analyzed. Video results explain when there are no analyzed frames for a threshold percentage. |
| Readability | Darkened secondary text and enlarged supporting labels, upload guidance, file details, results descriptions and connection controls. Long model values wrap beside their labels. |
| Accessibility | Main controls and disclosure rows have a minimum 44px target height. Keyboard focus is visible. The visible Browse and Replace buttons remain keyboard accessible without duplicate invisible file-input tab stops. |
| Responsive layout | Kept the warm off-white, charcoal and sage two-column desktop layout. Stack panels at 800px and below. Mobile stacks the header and navigation, tightens hero spacing and shortens the empty results area. |

## Verification

Tested localhost:5175 through its configured same-origin API proxy in headless Edge. Approved validation assets were selected from the checksum-verified development manifest. The image displayed 23.2% real / 76.8% fake. Video showed 73.5% average fake score, 12/12 analyzed frames, 12 faces and zero decoding failures. This verifies integration, not accuracy.

Exercised image selection, repeated selection, drag/drop, preview, prediction, invalid replacement, video preview, processing and results, frame disclosure, About and keyboard focus. Checked document overflow at 1440, 1366, 768 and 390 CSS pixels with the initial screen and expanded video results; none detected. No uncaught JS exceptions or console.error calls in the completed run. All 13 frontend tests and 155 backend tests passed; production build passed. One existing Starlette/httpx deprecation warning remains.

## Screenshots

[Desktop](phase13/initial-1440.png) · [Laptop](phase13/initial-1366.png) · [Tablet](phase13/initial-768.png) · [Mobile](phase13/initial-390.png)

[Image preview](phase13/image-upload.png) · [Image result](phase13/image-result.png) · [Upload error](phase13/error.png)

[Video preview](phase13/video-upload.png) · [Processing](phase13/video-processing.png) · [Video result](phase13/video-result.png) · [Mobile result](phase13/mobile-result.png)

Reproduce using backend/scripts/audit_ui.py with the existing backend virtual environment, local services and installed Edge. The script uses a temporary browser profile.

## Known limitations

Viewport emulation does not replace physical devices, native file-picker testing or screen-reader review. Focus and contrasting text were checked; no full WCAG certification is claimed. Insufficient analysis is covered by component and backend tests; the captured live video had sufficient faces. Codec preview support varies by browser. The average video score is uncalibrated and cannot establish authenticity. No model weights or training data were modified.
