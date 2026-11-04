"""Variáveis extras para reduzir o atraso na detecção da degradação.

O diagnóstico anterior mostrou que o modelo não sabe se um motor degrada rápido ou devagar e
"regride à média" (viés por motor correlacionado com a vida: r ~ -0,65). Estas variáveis dão
evidência sobre isso:

  - desvio em relação ao estado inicial (`{s}_dev`, `{s}_dev{w}`): cada motor começa com um nível
    de desgaste um pouco diferente; subtrair o nível inicial dele remove essa diferença entre motores.
  - inclinação (`{s}_slope{w}`): quão rápido o sensor está mudando (tendência da regressão linear
    nos últimos w ciclos).
  - número do ciclo (`cycle`): quanto tempo o motor já operou.

Tudo é CAUSAL (sem vazamento): a linha do ciclo t usa só ciclos <= t do MESMO motor.
O nível inicial usa os primeiros `n_baseline` ciclos: no ciclo t <= n é a média dos ciclos 1..t;
depois disso fica fixo na média dos n primeiros.
"""

import numpy as np
import pandas as pd

from cmapss.features import add_rolling_features, feature_columns

N_BASELINE = 10
SLOPE_WINDOWS = (10, 30)


def baseline_levels(df: pd.DataFrame, sensors: list[str], n_baseline: int = N_BASELINE) -> pd.DataFrame:
    """Nível inicial de cada sensor por motor (causal). `df` deve estar ordenado por unit e cycle."""
    em = df.groupby("unit")[sensors].expanding().mean().reset_index(level=0, drop=True).reindex(df.index)
    after = np.broadcast_to((df["cycle"].to_numpy() > n_baseline)[:, None], em.shape)
    em = em.mask(after)
    return em.groupby(df["unit"]).ffill()


def add_deviation_features(
    df: pd.DataFrame, sensors: list[str], windows: tuple[int, ...], n_baseline: int = N_BASELINE
) -> pd.DataFrame:
    """Acrescenta `{s}_dev` (sensor - nível inicial) e `{s}_dev{w}` (média móvel - nível inicial).

    Exige que as colunas `{s}_mean{w}` já existam (veja `add_rolling_features`).
    """
    base = baseline_levels(df, sensors, n_baseline)
    new = {}
    for s in sensors:
        new[f"{s}_dev"] = df[s] - base[s]
        for w in windows:
            new[f"{s}_dev{w}"] = df[f"{s}_mean{w}"] - base[s]
    return pd.concat([df, pd.DataFrame(new, index=df.index)], axis=1)


def _slope_fn(w: int):
    xc = np.arange(w) - (w - 1) / 2.0
    denom = float(np.dot(xc, xc))
    return lambda a: float(np.dot(xc, a) / denom)


def add_slope_features(
    df: pd.DataFrame, sensors: list[str], windows: tuple[int, ...] = SLOPE_WINDOWS
) -> pd.DataFrame:
    """Acrescenta `{s}_slope{w}`: inclinação (ciclo a ciclo) nos últimos w ciclos.

    Nos primeiros w-1 ciclos de cada motor a janela está incompleta e o valor fica NaN
    (o XGBoost trata NaN como "ausente" nativamente).
    """
    new = {}
    for w in windows:
        r = (
            df.groupby("unit")[sensors]
            .rolling(window=w, min_periods=w)
            .apply(_slope_fn(w), raw=True)
            .reset_index(level=0, drop=True)
            .reindex(df.index)
        )
        for s in sensors:
            new[f"{s}_slope{w}"] = r[s]
    return pd.concat([df, pd.DataFrame(new, index=df.index)], axis=1)


def build_features(
    df: pd.DataFrame,
    sensors: list[str],
    windows: tuple[int, ...],
    slope_windows: tuple[int, ...] = SLOPE_WINDOWS,
    n_baseline: int = N_BASELINE,
) -> pd.DataFrame:
    """Calcula todas as variáveis (rolling, desvio e inclinação). Depois é só escolher as colunas."""
    df = add_rolling_features(df, sensors, windows)
    df = add_deviation_features(df, sensors, windows, n_baseline)
    return add_slope_features(df, sensors, slope_windows)


def feature_columns_v2(
    sensors: list[str],
    windows: tuple[int, ...],
    slope_windows: tuple[int, ...] = SLOPE_WINDOWS,
    dev: bool = False,
    slope: bool = False,
    cycle: bool = False,
) -> list[str]:
    cols = feature_columns(sensors, windows, with_rolling=True)
    if dev:
        cols += [f"{s}_dev" for s in sensors]
        cols += [f"{s}_dev{w}" for w in windows for s in sensors]
    if slope:
        cols += [f"{s}_slope{w}" for w in slope_windows for s in sensors]
    if cycle:
        cols.append("cycle")
    return cols


# Todas as combinações de variáveis extras (ablação completa: 2^3 = 8 conjuntos)
VARIANTS = {
    "base": dict(dev=False, slope=False, cycle=False),
    "+normalização": dict(dev=True, slope=False, cycle=False),
    "+inclinação": dict(dev=False, slope=True, cycle=False),
    "+ciclo": dict(dev=False, slope=False, cycle=True),
    "+norm+incl": dict(dev=True, slope=True, cycle=False),
    "+norm+ciclo": dict(dev=True, slope=False, cycle=True),
    "+incl+ciclo": dict(dev=False, slope=True, cycle=True),
    "+norm+incl+ciclo": dict(dev=True, slope=True, cycle=True),
}
