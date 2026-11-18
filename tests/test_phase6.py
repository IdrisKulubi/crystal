import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from src.significance import paired_bootstrap


def _frame(y_true, y_pred, groups, seed=0):
    return pd.DataFrame({"clip_id": [f"c{i}" for i in range(len(y_true))], "group": groups, "seed": seed,
                         "y_true": y_true, "y_pred": y_pred})


def test_observed_delta_matches_sklearn():
    rng = np.random.default_rng(1)
    labs = np.array(["a", "b", "c"])
    y = labs[rng.integers(0, 3, 120)]
    pa = np.where(rng.random(120) < 0.7, y, labs[rng.integers(0, 3, 120)])
    pb = labs[rng.integers(0, 3, 120)]
    g = [f"g{i // 4}" for i in range(120)]
    a = pd.concat([_frame(y, pa, g, 0), _frame(y, pa, g, 1)])
    b = pd.concat([_frame(y, pb, g, 0), _frame(y, pb, g, 1)])
    res = paired_bootstrap(a, b, resamples=400)
    want = f1_score(y, pa, average="macro", zero_division=0) - f1_score(y, pb, average="macro", zero_division=0)
    assert abs(res["delta_macro_f1"] - want) < 1e-12
    assert res["ci95"][0] < res["delta_macro_f1"] < res["ci95"][1]
    assert res["p_value"] < 0.05


def test_identical_runs_give_zero_delta_and_p_one():
    y = ["a", "b", "a", "b", "a", "c"] * 5
    g = [f"g{i}" for i in range(30)]
    a = _frame(y, y[::-1], g)
    res = paired_bootstrap(a, a.copy(), resamples=200)
    assert res["delta_macro_f1"] == 0.0
    assert res["ci95"] == [0.0, 0.0]
    assert res["p_value"] == 1.0


def test_mismatched_clips_raise():
    y = ["a", "b"] * 5
    a = _frame(y, y, [f"g{i}" for i in range(10)])
    b = a.iloc[:-1]
    try:
        paired_bootstrap(a, b, resamples=10)
    except ValueError:
        return
    raise AssertionError("expected ValueError")


def test_load_outputs_and_figures(tmp_path):
    from src.selective import load_outputs, plot_reliability, plot_risk_coverage
    labels = ["x", "y"]
    rng = np.random.default_rng(0)
    for f in range(2):
        d = tmp_path / f"fold_{f}_seed_0"
        d.mkdir()
        logits = rng.normal(size=(20, 2)) * 3
        p = np.exp(logits / 2) / np.exp(logits / 2).sum(1, keepdims=True)
        df = pd.DataFrame({"y_true": [labels[i] for i in rng.integers(0, 2, 20)], "p_x": p[:, 0], "p_y": p[:, 1]})
        df.to_csv(d / "predictions.csv", index=False)
        np.save(d / "test_logits.npy", logits)
    o = load_outputs(tmp_path)
    assert o["labels"] == labels and o["p_raw"].shape == (40, 2) and o["p_cal"].shape == (40, 2)
    assert np.allclose(o["p_raw"].argmax(1), o["p_cal"].argmax(1))
    plot_risk_coverage([("m", o)], tmp_path / "fig2")
    plot_reliability("m", o, tmp_path / "fig3")
    assert (tmp_path / "fig2.png").is_file() and (tmp_path / "fig3.pdf").is_file()
