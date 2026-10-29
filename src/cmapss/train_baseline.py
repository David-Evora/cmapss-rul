"""Baseline: Regressão Linear e Random Forest para prever a RUL.

Uso:
    python -m cmapss.train_baseline --data-dir data
(rode a partir da pasta cmapss-rul com PYTHONPATH=src, ou instale com `pip install -e .`)
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

from cmapss.data import SENSORS, last_cycle_rows, load_test, load_train


def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def usable_sensors(train: pd.DataFrame) -> list[str]:
    """Remove sensores constantes (não carregam informação)."""
    return [s for s in SENSORS if train[s].nunique() > 1]


def run(data_dir: str, out_dir: str = "results") -> pd.DataFrame:
    train = load_train(data_dir)
    test, rul_true = load_test(data_dir)
    features = usable_sensors(train)

    x_train, y_train = train[features], train["RUL"]
    x_test = last_cycle_rows(test)[features]

    models = {
        "LinearRegression": LinearRegression(),
        "RandomForest": RandomForestRegressor(n_estimators=200, n_jobs=-1, random_state=42),
    }
    rows = []
    for name, model in models.items():
        model.fit(x_train, y_train)
        pred = model.predict(x_test)
        rows.append({"modelo": name, "rmse_teste": round(rmse(rul_true, pred), 2)})

    result = pd.DataFrame(rows)
    Path(out_dir).mkdir(exist_ok=True)
    result.to_csv(Path(out_dir) / "baseline.csv", index=False)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data")
    args = parser.parse_args()
    print(run(args.data_dir).to_string(index=False))
