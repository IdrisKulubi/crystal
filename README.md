# CRYSTAL — calibrated, leak-free infant cry analysis

**CRYSTAL** (CRY Self-supervised Transformer with Abstention Learning) is a two-stage infant-cry
system: (1) detect cries in household audio, (2) predict the likely reason (hungry,
pain/discomfort, tired, burping) with calibrated confidence, abstaining when unsure. It is
evaluated against a reproduced [DeepInfant](https://github.com/skytells-research/DeepInfant)
baseline under a strict baby-grouped protocol.

> Status: Phase 0 (infrastructure) done. See `docs/STATUS.md`.

## Setup

```bash
python -m venv .venv
.venv/Scripts/activate        # Windows  (Linux/macOS: source .venv/bin/activate)
pip install -r requirements.txt
pytest -q
```

## Reproducing everything (filled in as phases are completed)

```bash
python -m src.prepare_data          # raw -> data/processed + data/metadata.csv        (Phase 1)
python -m src.make_splits           # -> data/metadata_folds.csv (+ leakage checks)     (Phase 1)
python -m src.dataset_stats         # -> results/dataset_stats.json (paper Table 1)      (Phase 1)
python -m src.train --config configs/exp/E10_ast_ft.yaml --fold all --seeds 0,1,2   (Phase 2+)
python -m src.calibrate --run <run_id>                                               (Phase 6)
python -m src.report                # -> results/RESULTS.md, paper_numbers.json, tables/
```

## How results are tracked

Every run writes a folder `results/runs/<run_id>/` (config, environment, per-fold logits,
predictions, metrics) and appends a line to `results/registry.jsonl`. `python -m src.report`
turns the registry into `results/paper_numbers.json` — the only source of numbers for the
paper. The full contract is in `docs/RESULTS_SPEC.md`.

## Data and privacy

Infant audio is sensitive. `data/` is git-ignored; no audio is committed or uploaded.
Download instructions for each dataset are added in Phase 1.

## Licence

Code: Apache-2.0. Reused DeepInfant code keeps its Apache-2.0 notice.
