import numpy as np
import pandas as pd

from cmapss.feature_experiment import run
from cmapss.features import add_rolling_features
from cmapss.features_v2 import (
    add_deviation_features,
    add_slope_features,
    build_features,
    feature_columns_v2,
)

from test_data import _fake_files


def _toy(n_cycles=10):
    rows = []
    for unit, mult in ((1, 10.0), (2, 1000.0)):
        for cycle in range(1, n_cycles + 1):
            rows.append({"unit": unit, "cycle": cycle, "s2": mult * cycle})
    return pd.DataFrame(rows)


def _dev(df, n_baseline=3):
    df = add_rolling_features(df, ["s2"], (3,))
    return add_deviation_features(df, ["s2"], (3,), n_baseline)


def test_deviation_uses_causal_baseline():
    out = _dev(_toy())
    u1 = out[out["unit"] == 1].set_index("cycle")
    assert np.isclose(u1.loc[1, "s2_dev"], 0.0)  # baseline = o próprio ciclo 1
    assert np.isclose(u1.loc[2, "s2_dev"], 5.0)  # baseline = média(10, 20) = 15 -> 20 - 15
    assert np.isclose(u1.loc[5, "s2_dev"], 30.0)  # baseline fixo = média(10, 20, 30) = 20 -> 50 - 20
    assert np.isclose(u1.loc[5, "s2_dev3"], 20.0)  # média móvel (40) - baseline (20)
    # o outro motor não contamina este
    u2 = out[out["unit"] == 2].set_index("cycle")
    assert np.isclose(u2.loc[5, "s2_dev"], 5000.0 - 2000.0)


def test_slope_matches_true_slope_and_nan_when_incomplete():
    df = add_rolling_features(_toy(), ["s2"], (3,))
    out = add_slope_features(df, ["s2"], (3,))
    u1 = out[out["unit"] == 1].set_index("cycle")
    assert u1.loc[[1, 2], "s2_slope3"].isna().all()
    assert np.allclose(u1.loc[3:, "s2_slope3"], 10.0)  # s2 = 10 * ciclo
    u2 = out[out["unit"] == 2].set_index("cycle")
    assert np.allclose(u2.loc[3:, "s2_slope3"], 1000.0)


def test_no_future_leakage():
    base = build_features(_toy(), ["s2"], (3,), (3,), 3)
    changed = _toy()
    changed.loc[(changed["unit"] == 1) & (changed["cycle"] == 10), "s2"] = 99999.0
    out = build_features(changed, ["s2"], (3,), (3,), 3)
    cols = ["s2_dev", "s2_dev3", "s2_slope3", "s2_mean3"]
    a = base[(base["unit"] == 1) & (base["cycle"] < 10)][cols].to_numpy(dtype=float)
    b = out[(out["unit"] == 1) & (out["cycle"] < 10)][cols].to_numpy(dtype=float)
    assert np.allclose(a, b, equal_nan=True)


def test_feature_columns_exist_and_grow():
    df = build_features(_toy(), ["s2"], (3,), (3,), 3)
    all_cols = feature_columns_v2(["s2"], (3,), (3,), dev=True, slope=True, cycle=True)
    assert set(all_cols) <= set(df.columns)
    base = feature_columns_v2(["s2"], (3,), (3,))
    assert len(all_cols) > len(base)


def test_feature_experiment_runs(tmp_path):
    _fake_files(tmp_path, n_units=8)
    out = tmp_path / "res"
    variants = {
        "base": dict(dev=False, slope=False, cycle=False),
        "+norm+incl+ciclo": dict(dev=True, slope=True, cycle=True),
    }
    summary = run(
        str(tmp_path), out_dir=str(out), seeds=1, windows=(5,), slope_windows=(5,), n_baseline=3,
        params={"n_estimators": 20, "learning_rate": 0.1, "max_depth": 3}, variants=variants,
    )
    assert list(summary["variaveis"]) == ["base", "+norm+incl+ciclo"]
    assert summary.loc[1, "n_variaveis"] > summary.loc[0, "n_variaveis"]
    assert (out / "feature_experiment.csv").exists()
