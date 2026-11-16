import math

import numpy as np
import pytest

from src.results import metrics as M

LABELS = ["a", "b", "c"]


def onehot(y, k=3, p=1.0):
    out = np.full((len(y), k), (1 - p) / (k - 1))
    out[np.arange(len(y)), y] = p
    return out


def test_softmax_rows_sum_to_one_and_temperature_flattens():
    logits = np.array([[2.0, 0.0, -1.0], [0.0, 0.0, 0.0]])
    p1 = M.softmax(logits)
    p5 = M.softmax(logits, temperature=5.0)
    assert np.allclose(p1.sum(1), 1) and np.allclose(p5.sum(1), 1)
    assert p5[0].max() < p1[0].max()          # higher T -> less confident
    assert np.allclose(p1[1], 1 / 3)
    assert (p1.argmax(1) == p5.argmax(1)).all()  # T never changes the argmax


def test_perfect_predictions():
    y = np.array([0, 1, 2, 0, 1, 2])
    m = M.classification_metrics(y, onehot(y), LABELS)
    assert m["accuracy"] == 1.0 and m["macro_f1"] == 1.0 and m["balanced_accuracy"] == 1.0
    assert m["ece_15_cal"] == pytest.approx(0.0)
    assert m["brier"] == pytest.approx(0.0)
    assert m["aurc"] == pytest.approx(0.0)
    assert m["confusion_matrix"] == [[2, 0, 0], [0, 2, 0], [0, 0, 2]]
    assert m["per_class"]["b"] == {"precision": 1.0, "recall": 1.0, "f1": 1.0, "support": 2}


def test_majority_baseline_accuracy_vs_macro_f1():
    # 8 x class a, 1 x b, 1 x c; always predict a -> high accuracy, low macro-F1
    y = np.array([0] * 8 + [1, 2])
    probs = np.tile([0.8, 0.1, 0.1], (10, 1))
    m = M.classification_metrics(y, probs, LABELS)
    assert m["accuracy"] == pytest.approx(0.8)
    f1_a = 2 * 0.8 * 1.0 / 1.8
    assert m["macro_f1"] == pytest.approx(f1_a / 3)   # absent classes count as 0
    assert m["balanced_accuracy"] == pytest.approx(1 / 3)


def test_macro_f1_counts_classes_absent_from_test():
    y = np.array([0, 0, 1, 1])
    m = M.classification_metrics(y, onehot(y), LABELS)
    assert m["per_class"]["c"]["support"] == 0
    assert m["macro_f1"] == pytest.approx(2 / 3)


def test_ece_hand_computed():
    # bin [0.6,0.667): conf .6 x2 (1 right)   -> |0.5-0.6| * 2/4
    # bin [0.933,1]:   conf 1.0 x2 (both right) -> 0
    y = np.array([0, 0, 0, 0])
    probs = np.array([[0.6, 0.4, 0.0], [0.4, 0.6, 0.0], [1.0, 0, 0], [1.0, 0, 0]])
    assert M.ece(y, probs, 15) == pytest.approx(0.5 * 0.1)


def test_ece_overconfident_model():
    y = np.array([0, 1, 0, 1])
    probs = np.tile([0.9, 0.1, 0.0], (4, 1))  # always 90% sure of class a, right half the time
    assert M.ece(y, probs) == pytest.approx(0.4)


def test_nll_and_brier_uniform():
    y = np.array([0, 1, 2])
    u = np.full((3, 3), 1 / 3)
    assert M.nll(y, u) == pytest.approx(math.log(3))
    assert M.brier(y, u) == pytest.approx(1 - 1 / 3)


def test_risk_coverage_and_aurc():
    conf = np.array([0.9, 0.8, 0.7, 0.6])
    correct = np.array([1, 1, 0, 0])
    cov, risk = M.risk_coverage_curve(conf, correct)
    assert np.allclose(cov, [0.25, 0.5, 0.75, 1.0])
    assert np.allclose(risk, [0, 0, 1 / 3, 0.5])
    expected = 0.25 * (0 + 0) / 2 + 0.25 * (0 + 1 / 3) / 2 + 0.25 * (1 / 3 + 0.5) / 2
    assert M.aurc(conf, correct) == pytest.approx(expected)


def test_aurc_better_ranking_is_lower():
    correct = np.array([1, 1, 0, 0])
    good = M.aurc(np.array([0.9, 0.8, 0.2, 0.1]), correct)
    bad = M.aurc(np.array([0.1, 0.2, 0.8, 0.9]), correct)
    assert good < bad


def test_threshold_for_coverage_and_selective_accuracy():
    val = np.array([0.9, 0.8, 0.7, 0.6])
    assert M.threshold_for_coverage(val, 0.5) == pytest.approx(0.8)
    assert M.threshold_for_coverage(val, 0.7) == pytest.approx(0.7)   # ceil(2.8)=3
    assert M.threshold_for_coverage(val, 1.0) == pytest.approx(0.6)
    acc, cov = M.selective_accuracy(np.array([0.95, 0.85, 0.5, 0.4]), np.array([1, 0, 1, 1]), 0.8)
    assert acc == pytest.approx(0.5) and cov == pytest.approx(0.5)
    acc, cov = M.selective_accuracy(np.array([0.1]), np.array([1]), 0.8)
    assert math.isnan(acc) and cov == 0.0


def test_precision_at_coverage_uses_given_thresholds_only():
    y = np.array([0, 1, 2, 0])
    probs = np.array([[0.9, 0.05, 0.05], [0.2, 0.7, 0.1], [0.5, 0.3, 0.2], [0.6, 0.2, 0.2]])
    m = M.classification_metrics(y, probs, LABELS)
    assert m["precision_at_coverage"]["0.9"] is None      # no val thresholds -> not computed
    m = M.classification_metrics(y, probs, LABELS, thresholds={"0.9": 0.6, "0.7": 0.7, "0.5": 0.85})
    assert m["precision_at_coverage"]["0.7"] == pytest.approx(1.0)
    assert m["realised_coverage"]["0.7"] == pytest.approx(0.5)
    assert m["precision_at_coverage"]["0.9"] == pytest.approx(1.0)  # conf .9,.7,.6 kept; all right


def test_top2():
    y = np.array([0, 1])
    probs = np.array([[0.4, 0.5, 0.1], [0.6, 0.1, 0.3]])
    assert M.top_k_accuracy(y, probs, 2) == pytest.approx(0.5)


def test_uncal_vs_cal_ece_reported_separately():
    y = np.array([0, 1, 0, 1])
    raw = np.tile([0.9, 0.1, 0.0], (4, 1))
    cal = np.tile([0.5, 0.5, 0.0], (4, 1))
    m = M.classification_metrics(y, cal, LABELS, probs_uncal=raw)
    assert m["ece_15_uncal"] == pytest.approx(0.4)
    assert m["ece_15_cal"] == pytest.approx(0.0)


def test_detection_metrics():
    y = np.array([1, 1, 0, 0])
    s = np.array([0.9, 0.4, 0.6, 0.1])
    d = M.detection_metrics(y, s, 0.5, false_alarms=3, hours=1.5)
    assert d["accuracy"] == 0.5 and d["recall"] == 0.5 and d["miss_rate"] == 0.5
    assert d["roc_auc"] == pytest.approx(0.75)
    assert d["false_alarms_per_hour"] == pytest.approx(2.0)
    assert M.detection_metrics(y, s)["false_alarms_per_hour"] is None


def test_shape_mismatch_raises():
    with pytest.raises(ValueError):
        M.classification_metrics(np.array([0, 1]), np.ones((3, 3)) / 3, LABELS)
