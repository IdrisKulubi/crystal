# DECISIONS

`date | decision | reason | evidence (run ids)`

- 2026-11-16 | Model name: CRYSTAL (stored only in `configs/project.yaml`) | Idris confirmed | —
- 2026-11-16 | Std in summaries is the sample std (ddof=1) over all fold × seed results | standard for reporting variability of a small number of runs; stated in paper | —
- 2026-11-16 | Brier score = mean over clips of Σ_k (p_k − onehot_k)² (multi-class Brier) | RESULTS_SPEC says "MSE vs one-hot"; this is the usual multi-class form | —
- 2026-11-16 | ECE bins are [i/15, (i+1)/15); confidence 1.0 goes in the last bin | RESULTS_SPEC §5 (15 equal-width bins) | —
- 2026-11-16 | Coverage threshold τ_c = the ⌈c·n⌉-th highest validation confidence; test keeps conf ≥ τ_c | thresholds chosen on validation only (test folds are sacred) | —
- 2026-11-16 | `unified_v1` provisional = hungry, pain_discomfort, tired, burping; final set decided in Phase 1 after inspecting counts | HANDOFF §5.3 | —
- 2026-11-16 | Custom PyTorch loop (not HF Trainer) planned for `src/train.py` | simpler to save val/test logits exactly per RESULTS_SPEC | —
