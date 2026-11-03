"""Carregamento do dataset NASA C-MAPSS (subconjunto FD001 por padrão)."""

from pathlib import Path

import numpy as np
import pandas as pd

COLUMNS = ["unit", "cycle", "op1", "op2", "op3"] + [f"s{i}" for i in range(1, 22)]
SENSORS = [f"s{i}" for i in range(1, 22)]
RUL_CAP = 125  # limite comum na literatura: acima disso o motor é considerado "saudável"


def _read(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, sep=r"\s+", header=None, names=COLUMNS)


def load_train(
    data_dir: str | Path, subset: str = "FD001", cap: int | None = RUL_CAP
) -> pd.DataFrame:
    """Lê o treino e cria as colunas de vida útil restante (em ciclos).

    - `RUL_raw`: a RUL real, sem limite (ciclos até a falha).
    - `RUL`: o alvo de treino, limitado a `cap` (padrão 125). Use `cap=None` para não limitar.
    """
    df = _read(Path(data_dir) / f"train_{subset}.txt")
    max_cycle = df.groupby("unit")["cycle"].transform("max")
    df["RUL_raw"] = max_cycle - df["cycle"]
    df["RUL"] = df["RUL_raw"] if cap is None else df["RUL_raw"].clip(upper=cap)
    return df


def load_test(
    data_dir: str | Path, subset: str = "FD001", cap: int | None = RUL_CAP
) -> tuple[pd.DataFrame, np.ndarray]:
    """Lê o teste e a RUL verdadeira do último ciclo de cada motor.

    Por padrão a RUL real é limitada a `cap` (125), como no treino. Use `cap=None` para a RUL real sem limite.
    """
    data_dir = Path(data_dir)
    df = _read(data_dir / f"test_{subset}.txt")
    rul_true = pd.read_csv(data_dir / f"RUL_{subset}.txt", header=None)[0].to_numpy()
    return df, (rul_true if cap is None else rul_true.clip(max=cap))


def last_cycle_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Uma linha por motor: o último ciclo observado (é o que o teste avalia)."""
    return df.sort_values(["unit", "cycle"]).groupby("unit").tail(1).reset_index(drop=True)
