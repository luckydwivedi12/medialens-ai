# MediaLens AI project rules

- Use type hints in Python.
- Use TypeScript for frontend code.
- Separate API, preprocessing, inference, and UI responsibilities.
- Avoid unnecessary dependencies.
- Keep the design minimal and consistent: off-white backgrounds, charcoal text,
  subtle borders, generous spacing, and responsive desktop-first layouts.
- Never present an AI prediction as definitive proof of authenticity.
- Never store datasets, user uploads, or model weights in Git.
- Test changes before declaring them complete.
- Do not add ML dependencies, download datasets, or train models without an
  explicit request for that phase.

## Verification

From `backend`, run `.venv\Scripts\python.exe -m pytest`.
From `frontend`, run `npm run build`.
