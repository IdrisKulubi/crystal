"""Tests for prepare_data filename parsing, dedup, and grouped splits (synthetic data only)."""
import numpy as np
import pandas as pd
import pytest

from src.audio import pcm16_sha256
from src.dedup import find_duplicates, logmel_features, resolve, shifted_cosine
from src.make_splits import (
    check_grouped, choose_val_groups, group_overlap, grouped_folds, load_assignment, random_folds, save_assignment,
)
from src.prepare_data import parse_filename

SR = 16000


# --------------------------------------------------------------------------- filenames
def test_parse_ios_filename():
    p = parse_filename("0D1AD73E-4C5E-45F3-85C4-9A3CB71E8856-1430742197-1.0-m-04-hu")
    assert p["uuid"] == "0d1ad73e-4c5e-45f3-85c4-9a3cb71e8856"
    assert p["platform"] == "ios" and p["timestamp_unix"] == 1430742197.0
    assert p["gender"] == "m" and p["age_bracket"] == 0 and p["reason_from_filename"] == "hungry"
    assert not p["is_generated"]


def test_parse_android_and_generated_filename():
    p = parse_filename("generated2_0c8f14a9-6999-485b-97a2-913c1cbf099c-1431028888092-1.7-f-26-sc")
    assert p["platform"] == "android" and p["timestamp_unix"] == pytest.approx(1431028888.092)
    assert p["is_generated"] and p["generated_tag"] == "generated2"
    assert p["age_desc"] == "2-6 months" and p["reason_from_filename"] == "scared"


def test_parse_rejects_other_names():
    assert parse_filename("baby_cry_001") is None


# --------------------------------------------------------------------------- dedup
def _cry(seed, seconds=7.0):
    """Synthetic 'cry': random harmonic tones with random amplitude envelope."""
    rng = np.random.default_rng(seed)
    t = np.arange(int(seconds * SR)) / SR
    f0 = rng.uniform(300, 600) * (1 + 0.1 * np.sin(2 * np.pi * rng.uniform(0.3, 2) * t))
    phase = 2 * np.pi * np.cumsum(f0) / SR
    env = np.abs(np.sin(2 * np.pi * rng.uniform(0.2, 1.0) * t + rng.uniform(0, 6)))
    return (0.3 * env * (np.sin(phase) + 0.5 * np.sin(2 * phase))).astype(np.float32)


def test_pcm_hash_exact():
    a = _cry(0)
    assert pcm16_sha256(a) == pcm16_sha256(a.copy())
    assert pcm16_sha256(a) != pcm16_sha256(a * 0.9)


def test_near_duplicate_detection_with_shift_and_noise():
    a, b = _cry(1), _cry(2)
    shifted = np.concatenate([np.zeros(int(0.2 * SR), np.float32), a])[: len(a)]
    noisy = shifted + 0.003 * np.random.default_rng(3).normal(size=len(a)).astype(np.float32)
    feats = np.stack([logmel_features(x) for x in (a, b, noisy)])
    sim = shifted_cosine(feats, max_shift=12)
    assert sim[0, 2] > 0.9            # shifted, noisy copy is caught
    assert sim[0, 1] < sim[0, 2]      # different clip is less similar
    assert np.allclose(sim, sim.T)


def test_resolve_merges_groups_and_keeps_one_per_dataset():
    meta = pd.DataFrame({
        "clip_id": ["d__a", "d__b", "x__a_copy", "d__c"],
        "dataset": ["d", "d", "x", "d"],
        "group_raw": ["g1", "g2", "g9", "g3"],
        "pcm_sha256": ["h1", "h1", "h1", "h3"],   # a, b and the copy are identical audio
        "is_generated": [False, False, False, False],
    })
    pairs = find_duplicates(meta, None, 0.9, 0)
    assert len(pairs) == 2 and set(pairs.kind) == {"exact"}
    out = resolve(meta, pairs, ["d"]).set_index("clip_id")
    assert out.loc["d__b", "is_duplicate"] and out.loc["d__b", "duplicate_of"] == "d__a"
    assert not out.loc["d__a", "is_duplicate"] and not out.loc["x__a_copy", "is_duplicate"]
    assert out.loc["d__a", "group"] == out.loc["d__b", "group"] == out.loc["x__a_copy", "group"]
    assert out.loc["d__c", "group"] == "g3"


# --------------------------------------------------------------------------- splits
def _fake_table(n_groups=60, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for g in range(n_groups):
        label = rng.choice(["hungry", "tired", "pain", "burp"], p=[0.7, 0.12, 0.12, 0.06])
        for i in range(rng.integers(1, 5)):
            rows.append({"clip_id": f"c{g}_{i}", "group": f"g{g}", "label": label})
    return pd.DataFrame(rows)


def test_grouped_folds_have_no_group_leakage():
    df = grouped_folds(_fake_table(), "label", 5, 42, 0.15)
    check_grouped(df, 5)  # raises on leakage
    assert set(df["fold"]) == set(range(5))
    assert group_overlap(df, 5)["mean"] == 0.0
    for k in range(5):
        val = df[df[f"split_f{k}"] == "val"]
        assert 0.05 < len(val) / len(df) < 0.3


def test_grouped_folds_are_deterministic():
    a = grouped_folds(_fake_table(), "label", 5, 42, 0.15)
    b = grouped_folds(_fake_table(), "label", 5, 42, 0.15)
    pd.testing.assert_frame_equal(a, b)


def test_check_grouped_catches_leak():
    df = grouped_folds(_fake_table(), "label", 5, 42, 0.15)
    leaky = df.copy()
    i = leaky.index[(leaky["fold"] == 0)][0]
    j = leaky.index[(leaky["fold"] == 1)][0]
    leaky.loc[j, "group"] = leaky.loc[i, "group"]
    with pytest.raises(AssertionError):
        check_grouped(leaky, 5)


def test_random_split_leaks_groups():
    df = random_folds(_fake_table(), "label", 5, 42, 0.15)
    assert group_overlap(df, 5)["mean"] > 0.1


def test_choose_val_groups_covers_classes_when_possible():
    y = np.array(["a"] * 40 + ["b"] * 4 + ["c"] * 4)
    groups = np.array([f"a{i}" for i in range(40)] + ["b1", "b1", "b2", "b2", "c1", "c1", "c2", "c2"])
    val = choose_val_groups(y, groups, 7, 0)
    assert {"a", "b", "c"} <= set(y[val])
    train = np.setdiff1d(np.arange(len(y)), val)
    assert {"a", "b", "c"} <= set(y[train])
    assert not set(groups[val]) & set(groups[train])


def test_committed_assignment_round_trip_and_mismatch(tmp_path):
    table = _fake_table()
    g = grouped_folds(table, "label", 5, 42, 0.15)
    path = tmp_path / "grouped.csv"
    save_assignment(g, path, "label", 5)
    back = load_assignment(table, path, "label", 5)
    pd.testing.assert_frame_equal(back.sort_values("clip_id").reset_index(drop=True),
                                  g.sort_values("clip_id").reset_index(drop=True), check_dtype=False)
    bad = table.copy()
    bad.loc[0, "group"] = "someone_else"
    with pytest.raises(SystemExit):
        load_assignment(bad, path, "label", 5)
    with pytest.raises(SystemExit):
        load_assignment(table.iloc[1:], path, "label", 5)
