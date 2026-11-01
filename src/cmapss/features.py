"""Engenharia de variáveis: estatísticas móveis por motor.

Ideia: em vez de olhar uma "foto" ruidosa de um único ciclo, o modelo vê também
a tendência recente de cada sensor (média e variação nos últimos ciclos).

Importante (sem vazamento): as janelas são CAUSAIS. A linha do ciclo t usa apenas
os ciclos <= t do MESMO motor, nunca o futuro nem outros motores.
"""

import pandas as pd

DEFAULT_WINDOWS = (5, 15, 30)


def add_rolling_features(
    df: pd.DataFrame,
    sensors: list[str],
    windows: tuple[int, ...] = DEFAULT_WINDOWS,
) -> pd.DataFrame:
    """Devolve uma cópia de `df` com média e desvio móveis de cada sensor, por motor.

    Colunas novas: `{sensor}_mean{w}` e `{sensor}_std{w}` para cada janela `w`.
    No começo da vida do motor a janela ainda é curta (min_periods=1); o desvio
    de 1 ponto é NaN e vira 0.
    """
    df = df.sort_values(["unit", "cycle"]).reset_index(drop=True).copy()
    grouped = df.groupby("unit")[sensors]

    new_cols = {}
    for w in windows:
        rolled = grouped.rolling(window=w, min_periods=1)
        mean = rolled.mean().reset_index(level=0, drop=True)
        std = rolled.std().reset_index(level=0, drop=True).fillna(0.0)
        for s in sensors:
            new_cols[f"{s}_mean{w}"] = mean[s]
            new_cols[f"{s}_std{w}"] = std[s]

    return pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)


def feature_columns(
    sensors: list[str],
    windows: tuple[int, ...] = DEFAULT_WINDOWS,
    with_rolling: bool = True,
) -> list[str]:
    """Nomes das colunas usadas pelo modelo (sensores brutos + estatísticas móveis)."""
    cols = list(sensors)
    if with_rolling:
        for w in windows:
            for s in sensors:
                cols += [f"{s}_mean{w}", f"{s}_std{w}"]
    return cols
