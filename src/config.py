"""Config loading: defaults.yaml <- exp/*.yaml <- CLI overrides (RESULTS_SPEC §2)."""
from __future__ import annotations

import copy
import re
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = REPO_ROOT / "configs"

REQUIRED_KEYS = [
    "experiment_id",
    "name",
    "paper.table",
    "paper.row",
    "task",
    "data.metadata",
    "data.label_set",
    "eval.folds",
    "eval.seeds",
]
VALID_TASKS = {"reason", "detection", "dapt"}


class _Loader(yaml.SafeLoader):
    """SafeLoader that also reads ``1e-4`` / ``5E-5`` as floats (PyYAML's YAML 1.1 rules
    otherwise return them as strings, which would silently break learning rates)."""


_Loader.add_implicit_resolver(
    "tag:yaml.org,2002:float",
    re.compile(r"""^(?:[-+]?(?:[0-9][0-9_]*)\.[0-9_]*(?:[eE][-+]?[0-9]+)?
                    |[-+]?(?:[0-9][0-9_]*)(?:[eE][-+]?[0-9]+)
                    |\.[0-9_]+(?:[eE][-+]?[0-9]+)?
                    |[-+]?\.(?:inf|Inf|INF)
                    |\.(?:nan|NaN|NAN))$""", re.X),
    list("-+0123456789."),
)


def yaml_load(text_or_stream: Any) -> Any:
    return yaml.load(text_or_stream, Loader=_Loader)  # noqa: S506 (SafeLoader subclass)


def load_yaml(path: str | Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml_load(f) or {}


def deep_merge(base: dict, override: dict) -> dict:
    """Return a new dict: ``override`` merged recursively onto ``base``."""
    out = copy.deepcopy(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def get_path(cfg: dict, dotted: str, default: Any = None) -> Any:
    cur: Any = cfg
    for part in dotted.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return cur


def parse_overrides(items: list[str] | None) -> dict:
    """``["train.lr=1e-4", "eval.folds=[0]"]`` -> nested dict (values parsed as YAML)."""
    out: dict = {}
    for item in items or []:
        key, _, raw = item.partition("=")
        if not _:
            raise ValueError(f"override must be key=value, got {item!r}")
        node = out
        parts = key.strip().split(".")
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = yaml_load(raw)
    return out


def validate_config(cfg: dict) -> None:
    missing = [k for k in REQUIRED_KEYS if get_path(cfg, k) in (None, "", [])]
    if missing:
        raise ValueError(f"config missing required keys: {missing}")
    if cfg["task"] not in VALID_TASKS:
        raise ValueError(f"task must be one of {sorted(VALID_TASKS)}, got {cfg['task']!r}")
    if cfg.get("parent_run") and not cfg.get("change_vs_parent"):
        raise ValueError("a run with parent_run must state change_vs_parent (one change at a time)")


def load_config(exp_path: str | Path, overrides: list[str] | None = None,
                defaults_path: str | Path | None = None) -> dict:
    defaults = load_yaml(defaults_path or CONFIG_DIR / "defaults.yaml")
    cfg = deep_merge(defaults, load_yaml(exp_path))
    cfg = deep_merge(cfg, parse_overrides(overrides))
    validate_config(cfg)
    return cfg


def load_project(path: str | Path | None = None) -> dict:
    return load_yaml(path or CONFIG_DIR / "project.yaml")
