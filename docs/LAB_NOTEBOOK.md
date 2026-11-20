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

## 2026-11-17 — Phase 3 linear probes E05–E08 (local CPU)
- Runs: E05__wav2vec2_base_probe__20261116-234352__2df0f80, E06__hubert_base_probe__20261116-235847__2df0f80,
  E07__ast_probe__20261117-001010__9b88bc7, E08__whisper_base_probe__20261117-002638__9b88bc7 (E07/E08 started after
  commit 9b88bc7, which changed only fine-tuning code; probe code identical to 2df0f80). Embedding extraction ≈5–15 min
  per encoder on CPU; cached in data/embeddings/ (git-ignored).
- Test macro-F1 (5 folds × 3 seeds; seeds identical because the probe is deterministic):
  wav2vec2 0.193 ± 0.025, HuBERT 0.184 ± 0.037, AST 0.209 ± 0.015, Whisper-enc 0.184 ± 0.018 (majority 0.182).
  Only AST recognises any minority class (discomfort recall 0.35); belly_pain and burping recall 0 for all.
- Diagnostic (validation only): mean val macro-F1 by layer peaks in the middle (w2v2 layer 6 0.204, HuBERT 7 0.223,
  AST 6 0.242, Whisper 2 0.208) — middle layers are a bit better than the last, but all are close to chance.
- Reading: frozen general-purpose/speech representations + a linear classifier do not separate Donate-a-Cry reasons
  for unseen babies. Next: fine-tuning E10–E13 on Kaggle (E10 parent = E07, E12 parent = E05).

## 2026-11-17 — Kaggle E10–E13, first attempt (results not recoverable)
- Interactive Kaggle session at commit 985b049: E10__ast_ft__20261117-133442__985b049 completed (test macro-F1 0.188 ± 0.029
  as printed by the trainer), E12__w2v2_ft__20261117-142706__985b049 completed (0.182); E11 and E13 FAILED at model
  construction: Kaggle's torchao 0.10 is rejected by peft ("only versions above 0.16.0 are supported").
- The output zip was not downloaded before the interactive session ended, so these run folders are lost; the numbers above
  are from the screen and are NOT used anywhere (CLAUDE.md rule: never re-type numbers). Observed behaviour: AST full FT
  memorises the training set by epoch ~5 (train loss → 0) while val macro-F1 stays ≈0.18; wav2vec2 full FT predicts
  mostly "hungry".
- Fix (68a9f07): notebook uninstalls torchao before installing peft. Re-run of all four as a saved version
  (Save Version → Save & Run All) so the output is kept.

## 2026-11-17 — Phase 3 fine-tuning E10–E13 (Kaggle T4, commit 5a3ceae)
Runs imported from Kaggle zip: E10__ast_ft__20261117-165906__5a3ceae, E11__ast_lora__20261117-174815__5a3ceae,
E12__w2v2_ft__20261117-183854__5a3ceae, E13__w2v2_lora__20261117-190008__5a3ceae (all completed, folds sha 14c58690).
(One earlier saved version failed in 20 s at the GPU assert because the re-imported notebook had no accelerator set.)
Observations (from registry/metrics.json; paper values come from paper_numbers.json):
- Macro-F1 mean: E10 0.188, E11 0.183, E12 0.182, E13 0.171. None beats the frozen AST probe E07 (0.209);
  E12 equals the majority baseline E00 (0.182).
- Per-class test recall (mean over 15): belly_pain and burping ~0 for all (E13 belly_pain 0.11);
  discomfort E10 0.08 vs E07 0.35. Fine-tuning pushes models back towards "hungry".
- Best epochs are early (mostly 1–5), i.e. val F1 peaks before the model learns minority classes; with
  ~5 minority clips per val fold the early-stopping signal is very noisy.
Interpretation: on Donate-a-Cry alone (heavily hungry-dominated, few babies per minority class), fine-tuning
does not help over a frozen probe. Best backbone so far: AST (best probe and best FT).

## 2026-11-18 — E14 AST full FT + attention pooling (Kaggle T4, commit 871fcbb)
Run E14__ast_ft_attnpool__20261117-201115__871fcbb, completed, parent E10.
Macro-F1 0.193 ± 0.018 vs E10 0.188 ± 0.029 (gap far inside the spread); calibrated ECE 0.128 vs 0.158.
Recall: belly_pain 0.03, burping 0, discomfort 0.10, hungry 0.89, tired 0.04. Best epochs later (1–8).
Still below the frozen AST probe E07 (0.209). Phase 3 conclusion: on Donate-a-Cry alone, class imbalance /
tiny minority classes dominate; architecture changes (FT, LoRA, pooling) do not move macro-F1.

## 2026-11-18 — Imbalance study E15–E17 (Kaggle T4, commit 08b176f; parent E14)
Runs: E15__ast_attn_balsampler__20261118-072958__08b176f, E16__ast_attn_focal__20261118-082542__08b176f,
E17__ast_attn_selloss__20261118-091113__08b176f — all completed.
Macro-F1: E14 0.193±0.018 | E15 balanced sampler 0.191±0.019 | E16 focal 0.193±0.026 | E17 select on weighted
val loss 0.179±0.033. Belly_pain and burping recall 0 in all three; discomfort 0.12/0.10/0.17 (E07 probe 0.35).
Calibrated ECE: E14 0.128 remains best.
Conclusion: none of the three standard imbalance fixes moves macro-F1. Combined with E10–E14 (training data is fit
but validation F1 stays ~chance), the limit is cross-baby generalisation with very few babies per minority class,
not the loss/sampling. Next lever is more data (Phase 4: CryCeleb DAPT, other labelled sets).

## 2026-11-18 — Phase 4 setup: CryCeleb DAPT corpus + trainer (commit 4b72e6a)
CryCeleb2023 terms accepted by Idris; downloaded to data/raw/cryceleb (git-ignored). Licence CC BY-NC-ND 4.0:
research use OK, the DAPT checkpoint must NOT be shared/released (keep Kaggle notebook + any HF repo private).
Corpus (results/dapt_stats.json): 6.51 h, 26,093 segments from 946 babies packed into 3,733 chunks of <= 8 s
(segments are single expirations, median 0.8 s); 0 exact duplicates, 0 overlap with labelled data.
src/dapt.py: Wav2Vec2ForPreTraining from facebook/wav2vec2-base, 4 s crops, mask p .65 span 10, lr 5e-5,
10k steps, batch 16; collapse guard on codevector perplexity. CPU smoke (4 steps): loss 3.98, perplexity 108.
E20 = E05 + DAPT weights (probe); E21 = E12 + DAPT init (full FT). Next: Kaggle notebooks/kaggle_dapt.ipynb.

## 2026-11-18 — Phase 4 results: D01 DAPT + E20/E21 (Kaggle T4, commit 72875a9 = 4b72e6a code + docs)
D01__w2v2_cryceleb__20261118-111057__72875a9: completed 10k steps in 93 min (0.56 s/step). Contrastive loss
3.87 -> ~2.85 per masked frame, codevector perplexity 109 -> 280 (no collapse). fig6_dapt_loss generated.
E20 DAPT probe: macro-F1 0.169±0.022 vs E05 0.193 (worse; discomfort recall 0.04 vs 0.07).
E21 DAPT FT: 0.181±0.003 vs E12 0.182 (same; hungry recall 0.99, i.e. majority behaviour); calibrated ECE
0.119 vs 0.172.
Interpretation: DAPT learned the cry domain (loss fell, codebook healthy) but the representation it learned
(CryCeleb = cry sounds of many babies, 6.5 h) does not separate cry *reasons* in Donate-a-Cry. Still limited
by the labelled set. T4 "− domain-adaptive pretraining" is computable but the delta is ~0.

## 2026-11-18 — Phase 7 setup: cry detector E40 (AST) / E41 (small CNN)
Detection table detect_v1 (results/detection_splits_report.json, folds sha 3e86b880…, committed splits/detect_v1.csv):
cry = Donate-a-Cry 457 clips (221 groups) + CryCeleb first chunk per baby 946 + ESC-50 crying_baby 40;
non_cry = 49 other ESC-50 classes 1,960 clips (1,506 source files). 5.56 h. Grouped 5-fold, stratified by source x label.
Design: 1 s frames, everything band-limited to 4 kHz (Donate-a-Cry is 8 kHz: otherwise sample rate leaks the label),
silent frames of cry clips (< -20 dB vs loudest frame) excluded; AST with 1.28 s input (pos-emb cut from the
12x101 grid); best epoch by val ROC-AUC, threshold = max val F1. Known confound: most positives are not ESC-50
while all negatives are -> by_source.esc50_recall (ESC-50 crying_baby) is the in-domain check to report.
False alarms/hour needs Idris's household recording (still pending). CPU debug runs of both completed.

## 2026-11-18 — Phase 7 results: E40 AST / E41 small-CNN cry detector (Kaggle T4, commit 7509d81)
5 folds x 3 seeds, all 15 cells completed. Frame level (1 s frames, test folds):
E40 AST: F1 0.987±0.003, ROC-AUC 0.999±0.001, precision 0.986, recall 0.987; clip F1 0.988 (max-score pooling).
E41 CNN (60k params): F1 0.969±0.006, ROC-AUC 0.995±0.002; clip F1 0.974. CPU 3.2 ms/frame vs AST 164 ms/frame.
Per source (frame recall): CryCeleb 0.999 / 0.994, Donate-a-Cry 0.979 / 0.936, ESC-50 crying_baby 0.70±0.23 /
0.55±0.23; ESC-50 non-cry false-positive rate 0.013 / 0.031.
Interpretation: headline numbers are inflated by the source confound (most positives are not ESC-50, every
negative is). The honest in-domain number is ESC-50 crying_baby recall: AST finds ~70% of those frames, with
high fold variance (only ~8 clips per test fold). Report it next to T6. False alarms/hour still needs the
household recording. AST threshold chosen on val varies widely (0.54±0.31): scores are near 0/1 on this data.

## 2026-11-18 — Phase 6: calibration + selective prediction for every T3 model (no new training)
New: src/significance.py (paired cluster bootstrap over groups, pooled OOF macro-F1 averaged over seeds,
10k resamples, rng 0), src/selective.py (fig2 risk-coverage, fig3 reliability before/after from
test_logits.npy vs calibrated predictions.csv), report T5b (ECE before/after, AURC, precision@90/70/50 for
every T3 run), figure_sources.json. Report auto-computes ours-vs-DeepInfant significance once E36 exists.
Findings (T5b, mean ± std over 15 cells):
* Temperature scaling (fit on ~53 val clips/fold) helps the over-confident fine-tuned nets a lot
  (E12 0.434->0.172, E14 0.225->0.128, E21 0.446->0.119) but HURTS already-calibrated models
  (E00 0.017->0.067, E02 0.044->0.085, E07 0.125->0.181): val sets are too small to fit T reliably.
* Selective prediction gives little: precision@50% coverage is 82-86% for the best models vs 83.6% for the
  majority baseline, because accuracy is dominated by "hungry". Best AURC: E21 0.144, E02 0.154.
  Abstention does not fix the minority classes; macro-F1-at-coverage would be the more honest view (not added:
  that would change the evaluation protocol -> ask Idris).
* Significance (provisional, vs E02): E14 +0.022 pooled macro-F1, CI [0.002, 0.041], p 0.030;
  E07 +0.036, CI [-0.006, 0.082], p 0.105. Pooled-OOF deltas differ from the per-cell mean deltas in T3.

## 2026-11-18 — Protocol addition: macro_f1_at_coverage (approved by Idris, before E36)
Definition (RESULTS_SPEC §5, src/results/metrics.py selective_macro_f1): macro-F1 on the kept test clips
(confidence >= tau_c chosen on validation), averaged over ALL 5 classes, zero_division=0, so abstaining on a
whole class scores 0 for it. Recomputed for existing runs from predictions.csv abstain@ columns (no
retraining); "1.0" reproduces stored macro_f1 exactly (checked E10, E14). New in T5 (column) and T5b.
Result: abstention does NOT help the minority classes. Macro-F1 stays ~0.17-0.21 at every coverage for every
model (majority 0.182 flat; E02 0.183 flat; E14 0.190/0.193/0.187; E21 0.182/0.183/0.185). Highest at 50%:
wav2vec2 probe 0.211 ± 0.056 (noisy). Confident predictions are mostly confident "hungry" predictions.

## 2026-11-18 — Phase 5 setup: E30 noise / E31 SpecAugment / E32 mixup / E34 context (parent E14)
src/augment.py, training batches only. Parent = E14 (AST full FT + attention pooling): best fine-tuned run,
the only one significantly above DeepInfant (p 0.030), best calibrated ECE among T3 at the time.
* E30 noise: ESC-50 49 non-baby classes, band-limited to 4 kHz (Donate-a-Cry is 8 kHz -> wide-band noise
  would be a train/test mismatch), SNR U[0,20] dB, p 0.5, added only over the clip's real length.
  Household noise not recorded yet -> ESC-50 only (HANDOFF said ESC-50 + household).
* E31 SpecAugment: 1 freq mask U[0,24] + 1 time mask U[0,96] on the AST fbank (AST ESC-50 recipe).
* E32 mixup alpha 0.4 on fbank inputs, loss mixed with the same lambda.
* E33 (focal vs weighted CE) = E16 (already E14 + focal, 0.193 vs 0.193): not repeated.
* E34 context: one-hot age bracket (5) + gender (2) from the filename, concatenated to the pooled embedding.
* E35 (seed ensemble) needs val logits for calibration, which runs do not save -> decide after E30-E34.
Decision rule (HANDOFF): keep a component for E36 only if mean val macro-F1 (model_selection.best_val_macro_f1)
beats E14's 0.211. Note val sets are ~53 clips/fold, so this signal is noisy; record the decision either way.

## 2026-11-19 — Phase 5 single-change runs (E30, E31, E32, E34) imported
Runs (Kaggle T4, git a2e30e2, folds sha 14c58690…, 5 folds × 3 seeds, all completed):
E30__ast_attn_noise__20261119-005947__a2e30e2, E31__ast_attn_specaug__20261118-230208__a2e30e2,
E32__ast_attn_mixup__20261118-235420__a2e30e2, E34__ast_attn_context__20261118-220809__a2e30e2.

Selection uses VALIDATION macro-F1 only (mean over 15 cells of model_selection.best_val_macro_f1):
| run | val macro-F1 | vs E14 0.2109 | decision |
|---|---|---|---|
| E16 focal | 0.2115 ± 0.039 | +0.0006 | keep (rule), negligible |
| E30 noise | 0.2223 ± 0.042 | +0.011 | keep |
| E31 SpecAugment | 0.2223 ± 0.048 | +0.011 | keep |
| E32 mixup | 0.2326 ± 0.052 | +0.022 | keep |
| E34 context | 0.2141 ± 0.044 | +0.003 | keep (rule), negligible |
All gains are smaller than one std across cells; the pre-registered rule ("keep if mean val macro-F1
improves") was applied as written, without picking by test score. T4 ablations will show which
components matter. Test macro-F1 (reported, not used for selection) in RESULTS.md.

E36 (final "Ours") = E14 + focal + noise + SpecAugment + mixup + context. Ablations E36-noFocal,
-noNoise, -noSpecAug, -noMixup, -noContext (T4 rows "− SpecAugment" and "− mixup" added to
project.yaml paper_rows). The T4 row "− domain-adaptive pretraining" stays unfilled: E36 is AST and
DAPT was done on wav2vec2, so that ablation is not defined — decision pending with Idris.
Report must be run with `.venv/Scripts/python` (system Python lacks sklearn/scipy).

## 2026-11-19 — E36 final model + T4 ablations imported (Kaggle T4, git 1c39f27)
Runs: E36__crystal__20261119-063159__1c39f27; E36-noMixup/-noNoise/-noSpecAug/-noContext/-noFocal
(…20261119-074124 / -084548 / -094740 / -110044 / -121954__1c39f27). All completed, 5 folds × 3 seeds.

| run | val macro-F1 | test macro-F1 |
|---|---|---|
| E14 (parent) | 0.2109 ± 0.039 | 0.193 ± 0.018 |
| E36 full | 0.2135 ± 0.047 | 0.194 ± 0.024 |
| − mixup | 0.2052 | 0.190 |
| − noise | 0.2226 | 0.197 |
| − SpecAugment | 0.2241 | 0.192 |
| − context | 0.2227 | 0.191 |
| − focal | 0.2131 | 0.197 |
Finding: the components do NOT stack. E36 val (0.2135) is below E32 alone (mixup, 0.2326); removing
noise, SpecAugment or context *raises* val. Every T4 delta on test is within ±0.003, far inside the
fold/seed std (~0.02–0.04): no single component has a measurable effect on this data (~53 val clips
per fold). Reported as is. E36 vs E02 (paired cluster bootstrap): +0.018, CI [0.002, 0.034], p 0.031.
Calibrated ECE E36 0.156 (E14 0.128).

E35 (seed ensemble, exploratory, from saved test_logits; not a paper number): E36 single 0.194 →
3-seed mean-softmax ensemble 0.196; E14 0.193 → 0.188. No gain; E35 not built as a run.

Decision pending with Idris: keep E36 as "Ours" (rule applied as pre-registered) vs. redefine.
Phase 8 prepared: E90-deepinfant (E02 recipe) and E90-ours (E36 recipe) with
data.metadata = data/metadata_randomsplit.csv (sha f910bddc…, asserted in the Kaggle notebook).
CPU debug run of E90-deepinfant confirmed the random-split file is used.
**Decision (Idris, 2026-11-19): E36 stays "Ours"** — the pre-registered keep rule is applied as written;
no re-selection among the noisy single-change val scores. E90-ours keeps the E36 recipe.

## 2026-11-20 — T4 "− domain-adaptive pretraining" row dropped (Idris)
E36 is AST; DAPT (D01) was done on wav2vec2, so removing DAPT from E36 is undefined. The DAPT effect is
reported in T3 instead (DAPT wav2vec2 rows vs the plain wav2vec2-base rows). Removed from configs/project.yaml paper_rows.T4.

## 2026-11-20 — Phase 8 leakage demo (E90) imported (Kaggle T4, git bb98343)
Runs: E90-deepinfant__leak_randomsplit_deepinfant__20261119-212432__bb98343,
E90-ours__leak_randomsplit_ours__20261119-215357__bb98343. Folds file = metadata_randomsplit.csv (sha f910bddc…).
In the random split 68.0% of test clips share a baby with a training clip.
| model | grouped (T3) | random split (T4) | change |
|---|---|---|---|
| DeepInfant repro | 0.182 ± 0.002 | 0.182 ± 0.001 | none |
| Ours (E36) | 0.194 ± 0.025 | 0.280 ± 0.058 | +0.086 |
DeepInfant collapses to the majority class (hungry) under both splits (balanced acc 0.199 both), so
leakage cannot inflate it. E36 is the model that actually learns per-baby cues: random split inflates
its macro-F1 by +0.086 (balanced acc 0.209 → 0.270). Note: T4 "Δ vs full" for the DeepInfant row is vs
E36, not vs DeepInfant's own grouped score; the Discussion must state the per-model comparison.

## 2026-11-20 — Phase 9 efficiency (src/efficiency.py -> results/efficiency.json, table Teff)
Why a new benchmark: per-run inference_ms_per_clip_cpu was measured on whichever machine trained the
run (mostly Kaggle CPUs, some local), so rows were not comparable. Now one machine (Idris's laptop:
Intel family 6 model 154, 12 logical CPUs, torch 2.15 CPU, 10 threads), architectures rebuilt
locally (weights do not change cost/size), 50 fold-0 test clips, batch 1, raw waveform -> logits
incl. feature extraction, 5 warm-ups; all threads and 1 thread. Params match the paper runs exactly.
| model | params M | MB fp32 | ms (all) | ms (1 thr) |
|---|---|---|---|---|
| DeepInfant E02 | 19.80 | 79.2 | 83 | 106 |
| AST probe E07 | 86.19 | 344.8 | 1729 | 4289 |
| wav2vec2 FT E12 | 94.38 | 377.6 | 797 | 1319 |
| Ours E36 | 86.34 | 345.4 | 1612 | 3481 |
| AST detector E40 (1 s) | 85.37 | 341.5 | 552 | 694 |
| CNN detector E41 (1 s) | 0.06 | 0.25 | 6.3 | 5.9 |
Reading: ours is ~19× slower and ~4× larger than DeepInfant on CPU; still ~1.6 s for a 7 s clip,
i.e. faster than real time on a laptop. Two-stage deployment: the 0.25 MB CNN detector gates the
heavy classifier, which then only runs on detected cries. AST probe > ours in ms because it returns
all hidden states and does not crop to 7 s. ONNX/TFLite export (optional) not done.
Laptop timings are noisy (thermal/background load): treat ±10–20% as noise.

## 2026-11-20 — Phase 10 start: src/fill_paper.py + drafts
- `python -m src.fill_paper` copies the template to paper/drafts/CRYSTAL_paper_filled.docx (template
  sha unchanged, asserted in tests) and fills 65 placeholders from paper_numbers.json; filled runs lose
  the yellow highlight, 56 stay yellow (writing tasks, author details, own-dataset rows, BEATs row,
  references to verify). Tables 3/4/5 body rows rebuilt from the report tables. paper/drafts/fill_report.md
  lists both.
- Results/Discussion drafts are templates (paper/drafts_src/*.md.tmpl) with {key} tokens rendered from
  paper_numbers.json, so no number is typed. Rendered to paper/drafts/results.md, discussion.md.
- report.py: T2.* implementation keys from the E36 run's config/env and D01 config; T4 leak rows get
  delta_vs_grouped (vs the same model's grouped score); role leak_random_deepinfant added.
- Honest framing in the drafts: Ours (0.194) is not the top test row (AST probe 0.209); majority
  baseline 0.182 = DeepInfant; gains are significant vs DeepInfant but small; the leakage result is the
  strongest finding.
- Not committed (public repo): paper/drafts/, paper/drafts_src/ — Idris decides.
- Template prose that no longer matches what was built (LoRA, DAPT in the final model, context MLP,
  random cropping, median filter, AudioSet/ICSD positives) needs rewriting by Idris.
