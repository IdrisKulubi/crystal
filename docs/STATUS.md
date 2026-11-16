# STATUS
Last updated: 2026-11-17 by Claude Code (Phase 2 done)

## Current phase: 3 (pretrained encoders) — Phase 2 baselines complete

## Next task:
Phase 3: implement pretrained-encoder linear probes E05–E09 (frozen embeddings + logistic regression),
debug locally, then run on Kaggle with notebooks/kaggle_train.ipynb (CONFIGS list) and import.

## Blockers / needs Idris:
- [x] Confirm Donate-a-Cry grouping rule (group = app-install UUID) — see LAB_NOTEBOOK 2026-11-16 Phase 1
- [x] Burping kept as its own class (Idris, 2026-11-16)
- [ ] Listening audit: fill `data/listen/listen_sheet.csv` (25 clips)
- [x] Ran E02 on Kaggle (2026-11-17)
- [ ] Request Baby Chillanto access (email INAOE) — pipeline will work without it
- [ ] Accept CryCeleb terms on Hugging Face (needed by Phase 4, DAPT)
- [x] Confirm model name — CRYSTAL (2026-11-16)
- [ ] Record household noise (~30 min) + long household recording (1–2 h) — needed by Phases 5 & 7
- [x] GitHub repo: github.com/IdrisKulubi/crystal
- [ ] Find arXiv endorser — deferred; no co-author for now (feedback sought after completion)

## Phase checklist
- [x] P0 bootstrap  - [x] P1 data  - [x] P2 baselines  - [ ] P3 encoders
- [ ] P4 DAPT  - [ ] P5 improvements  - [ ] P6 calibration  - [ ] P7 detector
- [ ] P8 leakage  - [ ] P9 efficiency  - [ ] P10 paper

## Notes
- No deadline: personal project, go as fast as is sensible (Interspeech date in HANDOFF is not binding).
- Local machine: Windows 11, Python 3.13, no GPU → local runs use `--debug`; real training on Kaggle/Colab.

## Unfilled paper numbers (from RESULTS.md):
97 of 146 (Table 1 + E00/E01/E02 rows filled) — see `results/RESULTS.md`.

## Deferred data
- ESC-50 (noise aug, detector negatives) — download in Phase 5/7.
- Donate-a-Cry raw upload buckets (712 .3gp + 416 .caf, unchecked) — candidate unlabelled audio for Phase 4 DAPT;
  must exclude every group that appears in the eval folds. Needs ffmpeg (installed).
