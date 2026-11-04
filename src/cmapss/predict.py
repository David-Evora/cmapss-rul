"""Inferência com o modelo final salvo (usada pela demo e pela publicação no Hugging Face).

Entrada: o histórico de UM motor (ou vários, com a coluna `unit`) no formato do C-MAPSS:
26 colunas sem cabeçalho (unit, cycle, op1..op3, s1..s21), começando no ciclo 1.
As variáveis são causais: a previsão de cada ciclo só usa o passado do mesmo motor.
"""

import json
from pathlib import Path

import pandas as pd
from xgboost import XGBRegressor

from cmapss.data import COLUMNS
from cmapss.features_v2 import build_features

MODEL_FILE = "rul_xgb.json"
META_FILE = "metadata.json"


def load_model(model_dir: str | Path) -> tuple[XGBRegressor, dict]:
    model_dir = Path(model_dir)
    meta = json.loads((model_dir / META_FILE).read_text(encoding="utf-8"))
    model = XGBRegressor()
    model.load_model(str(model_dir / MODEL_FILE))
    return model, meta


def read_history(path: str | Path) -> pd.DataFrame:
    """Lê um histórico em .txt/.csv (separado por espaço, vírgula ou ponto e vírgula, sem cabeçalho)."""
    df = pd.read_csv(path, sep=r"[\s,;]+", header=None, engine="python")
    df = df.dropna(axis=1, how="all")  # separadores no começo/fim da linha criam colunas vazias
    if df.shape[1] != len(COLUMNS):
        raise ValueError(
            f"Esperava {len(COLUMNS)} colunas (unit, cycle, op1-3, s1-s21) e encontrei {df.shape[1]}."
        )
    df.columns = COLUMNS
    return df


def predict_trajectory(df: pd.DataFrame, model: XGBRegressor, meta: dict) -> pd.DataFrame:
    """RUL prevista em cada ciclo de cada motor (colunas: unit, cycle, rul_prevista)."""
    feats = build_features(
        df,
        meta["sensors"],
        tuple(meta["windows"]),
        tuple(meta["slope_windows"]),
        meta["n_baseline"],
    )
    out = feats[["unit", "cycle"]].copy()
    out["rul_prevista"] = model.predict(feats[meta["features"]])
    return out


def predict_last_cycle(df: pd.DataFrame, model: XGBRegressor, meta: dict) -> pd.DataFrame:
    """RUL prevista no último ciclo observado de cada motor."""
    traj = predict_trajectory(df, model, meta)
    return traj.sort_values(["unit", "cycle"]).groupby("unit").tail(1).reset_index(drop=True)
