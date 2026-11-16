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

## 2026-11-16 — Phase 2: E00 majority, E01 MFCC+SVM
Why: lower bounds under the grouped protocol (label set donateacry_v1, 5 folds × 3 seeds).
What: `src/train.py` (generic trainer; temperature scaling + coverage thresholds on val for every run),
`src/models/baselines.py`.
Result (test, mean ± std over 15): E00 acc 83.6 ± 1.6 %, macro-F1 0.182 ± 0.002, bal-acc 20.0 %.
E01 acc 77.8 ± 8.4 %, macro-F1 0.175 ± 0.010, bal-acc 18.6 ± 2.2 %.
Observations:
- First E01 run (…185937__bbaf546) was a bug: logits came from SVC Platt probabilities
  (`probability=True`), whose argmax collapsed to "hungry" for every clip although the grid search
  used `predict()`. Fixed to one-vs-rest decision scores (`break_ties=True` so predict == argmax).
  Old run kept and marked superseded in the registry.
- After the fix E01 is still ≈ majority: on validation no (C, γ) beats the majority-equivalent
  macro-F1 (~0.18). Hand-crafted MFCC statistics do not separate cry reasons on this data.
  This is a genuine (negative) result, not tuned further.
- Majority class already gets 83.6 % accuracy → DeepInfant's reported 89 % is only ~5 points above
  "always hungry" (different data/split, so not directly comparable; discuss neutrally).

## 2026-11-16 — Phase 2: E02 DeepInfant port
What: `src/models/deepinfant_repro.py` from DeepInfant `train.py` @ 7526568.
Observations:
- The published `train.py` cannot run: `conv_layers` ends with the SE gate (AdaptiveAvgPool→1×1
  convs→Sigmoid), which replaces the feature map with a [B,256,1,1] tensor, so the reshape to
  256×10 fails; `transform=True` makes the dataset call `True(mel)` (TypeError); the permute is
  frequency-major despite the "(batch, time, …)" comment; best checkpoint is chosen on the test split.
  The released 89 % model is a Core ML file, so this script is not what produced it.
  Minimal fixes D1–D5 documented in the module docstring and DECISIONS.md.
- Spectrograms computed with a PyTorch re-implementation of their librosa pipeline
  (max |Δ| = 0.0004 dB on 20 real clips; unit-tested) so training runs on GPU.
- CPU speed: ~4 s per batch of 32 → ~38 min per fold×seed → ~9.5 h for 15 runs on this laptop.
  Full E02 to run on Kaggle (`notebooks/kaggle_train.ipynb`, results imported with `src.import_runs`).

## 2026-11-17 — Kaggle split mismatch (caught by the notebook's guard)
- First Kaggle run stopped before training: same 1511 metadata rows and 996 duplicate pairs, but
  `make_splits` gave folds sha 9a3519aa… (with 4 test folds missing a rare class) instead of our 14c58690….
  Cause: different scikit-learn version on Kaggle (local 1.9.1) → different StratifiedGroupKFold shuffle.
- Fix: the fold assignment (clip_id, group, label, fold, split_f*) is now committed in `splits/` and
  re-applied everywhere; local re-application reproduces 14c58690… byte-for-byte. No runs were affected.

## 2026-11-17 — E02 DeepInfant reproduced (Kaggle GPU)
- Run E02__deepinfant_repro__20261116-220106__428232e, commit 428232e, folds sha 14c58690… (matches local).
  15/15 fold×seed completed, ~2.2 s/epoch on Kaggle GPU; imported with `src.import_runs`.
- Result: accuracy 83.3 ± 2.0 %, macro-F1 0.182 ± 0.002, balanced acc 19.9 ± 0.3 % — the same as the
  majority baseline. 14/15 runs selected epoch 1 (the original criterion, best val ACCURACY, is maximised
  by predicting "hungry" for everything); recall is 0 for every non-hungry class.
- Diagnostic only (not a paper number, validation data): the best val macro-F1 over epochs was 0.18–0.27,
  usually at epochs 35–50, so even under a macro-F1 criterion the CNN-BiLSTM learns little from 313
  training clips. Train loss stayed ≈0.33–0.45 (it does not even fit the training set closely).
- Interpretation: under a leak-free, baby-grouped protocol, DeepInfant's recipe does not beat "always hungry".
  Its reported 89 % accuracy is ≈5 points over the majority rate (83.6 %) and came from a clip-level random
  split (68 % of test clips share an app install with training — see splits_report.json).
- Possible later ablation (one change vs E02): early-stop on val macro-F1 and/or class-weighted CE, to show
  the failure is not only the selection criterion. Not run yet.
