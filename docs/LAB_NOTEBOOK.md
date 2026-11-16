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

## 2026-11-16 — Phase 1 data
Why: build a leak-free benchmark before any model is trained.
What: cloned Donate-a-Cry (cleaned set) and DeepInfant; `prepare_data` (16 kHz mono + metadata),
`dedup` (exact PCM hash + shift-tolerant log-mel cosine ≥ 0.90), `make_splits`
(StratifiedGroupKFold 5 folds, seed 42; ~15% of groups per training portion for validation),
`dataset_stats` (+ fig5), `listen_check` (25 clips for audit). 44 tests pass.
Results (all in results/dataset_stats.json, results/splits_report.json):
- Donate-a-Cry: 457 clips, 221 groups (app installs), 52.7 min, all clips ≈7 s, **all 8 kHz**
  (so content only up to 4 kHz; models see upsampled audio). Classes: hungry 382 (83.6%),
  discomfort 27, tired 24, belly_pain 16, burping 8. belly_pain and burping each come from only 7 installs;
  one install alone has 8 belly_pain clips recorded seconds apart.
- discomfort folder contains 7 clips whose filename code is `ch` (cold/hot): the corpus maintainers
  merged cold/hot into discomfort. Kept as discomfort.
- No exact or near duplicates inside Donate-a-Cry (even at cosine 0.90).
- 24 installs have clips under more than one label.
- **DeepInfant overlap:** `Data/` = all 457 Donate-a-Cry clips byte-for-byte (PCM). `Data/v2` (597 clips)
  contains 469 clips duplicating Donate-a-Cry (430 exact + re-encoded 44.1 kHz copies) and 24 synthetic
  `generatedN_` copies of other clips. DeepInfant's `prepare_dataset.py` splits `Data/v2` with
  `train_test_split(test_size=0.2, random_state=42)` per class at clip level, so augmented copies and
  same-install clips can fall on both sides. To be stated neutrally in the paper.
- DeepInfant's current `train.py` uses n_fft 1024, hop 256, 80 mels (README says 2048/512/128) → E02
  follows the code; deviation noted.
- Random clip-level split: 68.0% of test clips have a same-install clip in training (grouped: 0%).
- Validation sets: first attempt left a rare class out of 3 validation sets → `choose_val_groups`
  now deterministically picks the first seeded candidate covering every class; 0 warnings.
Observations: tiny minority classes (burping 1–2 test clips per fold) → expect very noisy per-class
scores; macro-F1 std will be large. Consider repeated CV later.
Next: Idris confirms grouping + burping decision; then Phase 2 baselines.
