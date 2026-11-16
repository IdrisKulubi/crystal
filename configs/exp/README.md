# Experiment configs

One YAML per experiment ID from HANDOFF §7, named `E<id>_<short_name>.yaml`
(e.g. `E10_ast_ft.yaml`). Each file only states what differs from `configs/defaults.yaml`
and must set `experiment_id`, `name`, `paper.table`, `paper.row`, and — unless it is a
baseline — `parent_run` + `change_vs_parent` (one change at a time).

The `paper.row` text must match a row in `configs/project.yaml -> paper_rows` exactly,
otherwise report.py will not place the result in the paper tables.

E04 ("DeepInfant reported", 89%) is a literature constant, stored in
`configs/project.yaml -> constant_rows`, not a run.
