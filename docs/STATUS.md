# STATUS
Last updated: 2026-11-16 by Claude Code (Phase 1)

## Current phase: 1 complete → next is Phase 2 (baselines)

## Next task:
Phase 2 — E00 majority, E01 MFCC+SVM, E02 DeepInfant reproduced (port `baselines/DeepInfant/train.py`
architecture into `src/models/deepinfant_repro.py`), generic `src/train.py`, Kaggle notebook.

## Blockers / needs Idris:
- [x] Confirm Donate-a-Cry grouping rule (group = app-install UUID) — see LAB_NOTEBOOK 2026-11-16 Phase 1
- [x] Burping kept as its own class (Idris, 2026-11-16)
- [ ] Listening audit: fill `data/listen/listen_sheet.csv` (25 clips)
- [ ] Request Baby Chillanto access (email INAOE) — pipeline will work without it
- [ ] Accept CryCeleb terms on Hugging Face (needed by Phase 4, DAPT)
- [x] Confirm model name — CRYSTAL (2026-11-16)
- [ ] Record household noise (~30 min) + long household recording (1–2 h) — needed by Phases 5 & 7
- [x] GitHub repo: github.com/IdrisKulubi/crystal
- [ ] Find arXiv endorser — deferred; no co-author for now (feedback sought after completion)

## Phase checklist
- [x] P0 bootstrap  - [x] P1 data  - [ ] P2 baselines  - [ ] P3 encoders
- [ ] P4 DAPT  - [ ] P5 improvements  - [ ] P6 calibration  - [ ] P7 detector
- [ ] P8 leakage  - [ ] P9 efficiency  - [ ] P10 paper

## Notes
- No deadline: personal project, go as fast as is sensible (Interspeech date in HANDOFF is not binding).
- Local machine: Windows 11, Python 3.13, no GPU → local runs use `--debug`; real training on Kaggle/Colab.

## Unfilled paper numbers (from RESULTS.md):
110 of 146 (Table 1 / dataset numbers now filled; all experiment numbers pending) — see `results/RESULTS.md`.

## Deferred data
- ESC-50 (noise aug, detector negatives) — download in Phase 5/7.
- Donate-a-Cry raw upload buckets (712 .3gp + 416 .caf, unchecked) — candidate unlabelled audio for Phase 4 DAPT;
  must exclude every group that appears in the eval folds. Needs ffmpeg (installed).
