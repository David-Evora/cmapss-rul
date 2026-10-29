"""Carregamento do dataset NASA C-MAPSS (subconjunto FD001 por padrão)."""

from pathlib import Path

import numpy as np
import pandas as pd

COLUMNS = ["unit", "cycle", "op1", "op2", "op3"] + [f"s{i}" for i in range(1, 22)]
SENSORS = [f"s{i}" for i in range(1, 22)]
RUL_CAP = 125  # limite comum na literatura: acima disso o motor é considerado "saudável"


def _read(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, sep=r"\s+", header=None, names=COLUMNS)


def load_train(data_dir: str | Path, subset: str = "FD001") -> pd.DataFrame:
    """Lê o treino e cria a coluna RUL (vida útil restante, em ciclos)."""
    df = _read(Path(data_dir) / f"train_{subset}.txt")
    max_cycle = df.groupby("unit")["cycle"].transform("max")
    df["RUL"] = (max_cycle - df["cycle"]).clip(upper=RUL_CAP)
    return df


def load_test(data_dir: str | Path, subset: str = "FD001") -> tuple[pd.DataFrame, np.ndarray]:
    """Lê o teste e o RUL verdadeiro do último ciclo de cada motor."""
    data_dir = Path(data_dir)
    df = _read(data_dir / f"test_{subset}.txt")
    rul_true = pd.read_csv(data_dir / f"RUL_{subset}.txt", header=None)[0].to_numpy()
    return df, rul_true.clip(max=RUL_CAP)


def last_cycle_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Uma linha por motor: o último ciclo observado (é o que o teste avalia)."""
    return df.sort_values(["unit", "cycle"]).groupby("unit").tail(1).reset_index(drop=True)
