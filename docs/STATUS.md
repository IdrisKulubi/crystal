# STATUS
Last updated: 2026-11-20 by Claude Code (Phase 10: fill_paper + drafts)

## Current phase: 10 in progress — filled draft + Results/Discussion drafts ready for review

## Next task:
Idris reviews paper/drafts/ (CRYSTAL_paper_filled.docx, results.md, discussion.md, fill_report.md) and rewrites
template prose that no longer matches. Then: anonymised Interspeech version, reference verification, README.

## Blockers / needs Idris:
- [x] Confirm Donate-a-Cry grouping rule (group = app-install UUID) — see LAB_NOTEBOOK 2026-11-16 Phase 1
- [x] Burping kept as its own class (Idris, 2026-11-16)
- [ ] Listening audit: fill `data/listen/listen_sheet.csv` (25 clips)
- [x] Ran E02 on Kaggle (2026-11-17)
- [ ] Request Baby Chillanto access (email INAOE) — pipeline will work without it
- [x] Accept CryCeleb terms on Hugging Face (2026-11-18)
- [x] Confirm model name — CRYSTAL (2026-11-16)
- [ ] Record household noise (~30 min) + long household recording (1–2 h) — needed by Phases 5 & 7
- [x] GitHub repo: github.com/IdrisKulubi/crystal
- [ ] Find arXiv endorser — deferred; no co-author for now (feedback sought after completion)

## Phase checklist
- [x] P0 bootstrap  - [x] P1 data  - [x] P2 baselines  - [x] P3 encoders
- [x] P4 DAPT  - [x] P5 improvements  - [x] P6 calibration  - [x] P7 detector
- [x] P8 leakage  - [x] P9 efficiency  - [ ] P10 paper

## Notes
- No deadline: personal project, go as fast as is sensible (Interspeech date in HANDOFF is not binding).
- Local machine: Windows 11, Python 3.13, no GPU → local runs use `--debug`; real training on Kaggle/Colab.

## Unfilled paper numbers (from RESULTS.md):
328 of 343 filled (T2 implementation keys added)

## Deferred data
- ESC-50: cloned for Phase 7 (data/raw/ESC-50, CC BY-NC); reuse for Phase 5 noise aug.
- Donate-a-Cry raw upload buckets (712 .3gp + 416 .caf, unchecked) — candidate unlabelled audio for Phase 4 DAPT;
  must exclude every group that appears in the eval folds. Needs ffmpeg (installed).
