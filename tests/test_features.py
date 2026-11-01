import numpy as np
import pandas as pd

from cmapss.features import add_rolling_features, feature_columns
from cmapss.train_features import run

from test_data import _fake_files


def _toy():
    rows = []
    for unit in (1, 2):
        for cycle in range(1, 11):
            rows.append({"unit": unit, "cycle": cycle, "s2": float(cycle * (10 if unit == 1 else 100))})
    return pd.DataFrame(rows)


def test_rolling_mean_uses_only_same_unit_and_past():
    df = _toy()
    out = add_rolling_features(df, ["s2"], windows=(3,))
    u1 = out[out["unit"] == 1].reset_index(drop=True)
    # ciclo 5 do motor 1: média dos ciclos 3,4,5 -> (30+40+50)/3 = 40
    assert np.isclose(u1.loc[4, "s2_mean3"], 40.0)
    # início da vida: janela curta (só o ciclo 1)
    assert np.isclose(u1.loc[0, "s2_mean3"], 10.0)
    # motor 2 não é contaminado pelo motor 1
    u2 = out[out["unit"] == 2].reset_index(drop=True)
    assert np.isclose(u2.loc[0, "s2_mean3"], 100.0)


def test_no_future_leakage():
    df = _toy()
    base = add_rolling_features(df, ["s2"], windows=(3,))
    changed = df.copy()
    changed.loc[(changed["unit"] == 1) & (changed["cycle"] == 10), "s2"] = 9999.0
    out = add_rolling_features(changed, ["s2"], windows=(3,))
    # mudar o ciclo 10 não pode alterar nenhuma linha anterior
    a = base[(base["unit"] == 1) & (base["cycle"] < 10)]["s2_mean3"].to_numpy()
    b = out[(out["unit"] == 1) & (out["cycle"] < 10)]["s2_mean3"].to_numpy()
    assert np.allclose(a, b)


def test_feature_columns_match_dataframe():
    df = _toy()
    out = add_rolling_features(df, ["s2"], windows=(3, 5))
    cols = feature_columns(["s2"], windows=(3, 5))
    assert set(cols) <= set(out.columns)
    assert not out[cols].isna().any().any()


def test_pipeline_runs(tmp_path):
    _fake_files(tmp_path, n_units=8)
    result = run(str(tmp_path), out_dir=str(tmp_path / "results"), fast=True)
    assert len(result) == 4
    assert set(result["variaveis"]) == {"sensores brutos", "+ médias móveis"}
    assert (result["rmse_teste"] >= 0).all()
