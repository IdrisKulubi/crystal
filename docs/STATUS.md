# STATUS
Last updated: 2026-11-16 by Claude Code

## Current phase: 0 complete → next is Phase 1 (data)

## Next task:
Phase 1 — download Donate-a-Cry + ESC-50, write `src/prepare_data.py`, `src/dedup.py`,
`src/make_splits.py`, `src/dataset_stats.py`; print 20 Donate-a-Cry filenames and confirm the
UUID grouping rule with Idris before relying on it (HANDOFF §5.2).

## Blockers / needs Idris:
- [ ] Request Baby Chillanto access (email INAOE) — pipeline will work without it
- [ ] Accept CryCeleb terms on Hugging Face (needed by Phase 4, DAPT)
- [x] Confirm model name — CRYSTAL (2026-11-16)
- [ ] Record household noise (~30 min) + long household recording (1–2 h) — needed by Phases 5 & 7
- [ ] Create the GitHub repo and tell Claude its URL (private recommended until the paper is out)
- [ ] Find arXiv endorser — deferred; no co-author for now (feedback sought after completion)

## Phase checklist
- [x] P0 bootstrap  - [ ] P1 data  - [ ] P2 baselines  - [ ] P3 encoders
- [ ] P4 DAPT  - [ ] P5 improvements  - [ ] P6 calibration  - [ ] P7 detector
- [ ] P8 leakage  - [ ] P9 efficiency  - [ ] P10 paper

## Notes
- No deadline: personal project, go as fast as is sensible (Interspeech date in HANDOFF is not binding).
- Local machine: Windows 11, Python 3.13, no GPU → local runs use `--debug`; real training on Kaggle/Colab.

## Unfilled paper numbers (from RESULTS.md):
All of them except `model_name` and the DeepInfant-reported 89% constant — see `results/RESULTS.md`.
