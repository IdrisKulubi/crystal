# LAB NOTEBOOK

Running log of what was tried, why, and what happened (including failures). Newest at the bottom.

## 2026-11-16 — Phase 0 bootstrap
Why: build the results-capture machinery before any experiment, so every number is traceable.
What: repo layout, configs (`project.yaml`, `defaults.yaml`), `src/config.py`,
`src/results/{run,metrics,aggregate}.py`, `src/report.py`, 33 unit tests.
Result: `pytest -q` passes; `python -m src.report` writes an empty RESULTS.md listing every
paper placeholder as unfilled (only `model_name` and the DeepInfant-reported 89% constant are filled).
Observations:
- Bug caught by tests: PyYAML reads `1e-4` as a *string* (YAML 1.1 rule). Added a loader
  (`src.config.yaml_load`) that parses scientific notation as floats; used everywhere configs are read.
- Local env is Python 3.13 (HANDOFF suggested 3.10–3.12); fine for Phase 0, will confirm
  torch/torchaudio/transformers install cleanly in Phase 1–2.
Next: Phase 1 (data).
