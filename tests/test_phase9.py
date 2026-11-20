import json

import numpy as np
import torch
from torch import nn

from src import efficiency as E
from tests.test_report import _report


def test_size_and_count():
    m = nn.Linear(10, 5)  # 55 params, fp32 -> ~220 bytes of tensor data + pickle overhead
    m.bias.requires_grad = False
    total, trainable = E.count(m)
    assert (total, trainable) == (55, 50)
    assert 55 * 4 / 1e6 <= E.size_mb(m) < 55 * 4 / 1e6 + 0.01
    assert abs(E.size_mb(m.half()) - E.size_mb(nn.Linear(10, 5))) < 1e-4  # always measured as fp32


def test_time_ms_counts_calls():
    calls = []
    out = E.time_ms(lambda w: calls.append(1), [np.zeros(3)], n=7, warmup=2)
    assert len(calls) == 9 and out["n"] == 7 and 0 <= out["median"] <= out["p90"]


def test_small_detector_builds_and_predicts():
    module, predict, override = E.build("cnn_detector", 5)
    assert override is None and E.count(module)[0] == 60465
    assert predict(np.random.default_rng(0).standard_normal(16000 * 2).astype(np.float32)).shape == (1,)
    assert predict(np.zeros(8000, np.float32)).shape == (1,)  # short input is padded to 1 s


def test_deepinfant_builds_and_predicts():
    module, predict, _ = E.build("deepinfant", 5)
    assert E.count(module)[0] == 19797141  # same as the E02 run's params_total
    with torch.no_grad():
        assert predict(np.zeros(16000 * 3, np.float32)).shape == (1, 5)


def test_report_efficiency_table(results_dir):
    rec = {"table": "T3", "row": "Ours (CRYSTAL)", "experiment_id": "E36", "params_total": 86338985,
           "params_trainable": 86338985, "size_mb_fp32": 345.4,
           "ms_cpu_all_threads": {"median": 1593.2, "p90": 1700.0, "n": 50},
           "ms_cpu_1_thread": {"median": 3027.8, "p90": 3100.0, "n": 50}}
    det = dict(rec, table="T6", row="Small CNN detector", experiment_id="E41", params_total=60465,
               params_trainable=60465, size_mb_fp32=0.25, ms_cpu_all_threads={"median": 3.7, "p90": 4.0, "n": 50},
               ms_cpu_1_thread={"median": 7.8, "p90": 8.0, "n": 50})
    results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / "efficiency.json").write_text(json.dumps(
        {"hardware": {"processor": "cpu", "cpu_count_logical": 4, "torch": "x"},
         "models": {"ours": rec, "cnn_detector": det}}), encoding="utf-8")
    res, pn, md = _report(results_dir)
    assert pn["eff.ours_crystal.params_m"] == "86.34"
    assert pn["eff.ours_crystal.cpu_ms"] == "1593" and pn["eff.small_cnn_detector.cpu_ms"] == "3.7"
    assert pn["eff.small_cnn_detector.size_mb"] == "0.25" and pn["eff.ours_crystal.size_mb"] == "345.4"
    assert (results_dir / "tables" / "Teff.md").is_file()
    assert "Efficiency — one machine" in md and "| Small CNN detector | 1 s frame |" in md


def test_report_without_efficiency_warns(results_dir):
    res, pn, md = _report(results_dir)
    assert any("efficiency.json" in w for w in res["warnings"])
