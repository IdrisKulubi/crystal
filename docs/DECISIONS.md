# DECISIONS

`date | decision | reason | evidence (run ids)`

- 2026-11-16 | Model name: CRYSTAL (stored only in `configs/project.yaml`) | Idris confirmed | —
- 2026-11-16 | Std in summaries is the sample std (ddof=1) over all fold × seed results | standard for reporting variability of a small number of runs; stated in paper | —
- 2026-11-16 | Brier score = mean over clips of Σ_k (p_k − onehot_k)² (multi-class Brier) | RESULTS_SPEC says "MSE vs one-hot"; this is the usual multi-class form | —
- 2026-11-16 | ECE bins are [i/15, (i+1)/15); confidence 1.0 goes in the last bin | RESULTS_SPEC §5 (15 equal-width bins) | —
- 2026-11-16 | Coverage threshold τ_c = the ⌈c·n⌉-th highest validation confidence; test keeps conf ≥ τ_c | thresholds chosen on validation only (test folds are sacred) | —
- 2026-11-16 | `unified_v1` provisional = hungry, pain_discomfort, tired, burping; final set decided in Phase 1 after inspecting counts | HANDOFF §5.3 | —
- 2026-11-16 | Custom PyTorch loop (not HF Trainer) planned for `src/train.py` | simpler to save val/test logits exactly per RESULTS_SPEC | —
- 2026-11-16 | Split group = Donate-a-Cry app-install UUID (lower-cased), merged across duplicate clusters | best available proxy for "same baby" (corpus README: UUID = app installation); stricter than session grouping | results/splits_report.json (PENDING Idris confirmation)
- 2026-11-16 | Benchmark = Donate-a-Cry cleaned set only (457 clips); DeepInfant Data/ and Data/v2 used only to measure overlap | Data/ is identical to Donate-a-Cry; v2 adds re-encoded and synthetic copies | dataset_stats.json → overlap
- 2026-11-16 | Folds stratified on 5 original Donate-a-Cry classes (donateacry_v1) | apples-to-apples with DeepInfant; unified_v1 derived from same folds | —
- 2026-11-16 | Near-duplicate rule: log-mel (32 mels, ≤4 kHz, 40 ms hop, 7 s) cosine ≥ 0.90, max over ±0.5 s shifts | Donate-a-Cry is 8 kHz; mean-pooled embeddings would flag different cries of the same baby; 0.90 is conservative (merging groups is the safe direction) | dedup_report.csv
- 2026-11-16 | Validation hold-out: first seeded StratifiedGroupKFold(7) candidate whose val covers every class with ≥2 groups and leaves all classes in train | avoids val sets missing a class (val macro-F1 would be biased); uses no test information | splits_report.json
- 2026-11-16 | Keep `discomfort` clips whose filename code is `ch` (7 clips) as discomfort | corpus maintainers' cleaned-folder label is the published label | —
