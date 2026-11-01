"""Dia 4: médias móveis + XGBoost, com validação separada por motor.

Compara, para Random Forest e XGBoost:
  - só sensores brutos
  - sensores + estatísticas móveis (média e desvio por motor)

Métricas:
  - rmse_cv:    validação cruzada AGRUPADA POR MOTOR no treino (nenhum motor aparece
                em treino e validação ao mesmo tempo -> sem vazamento). É a métrica para
                ESCOLHER o modelo.
  - rmse_teste: erro no último ciclo de cada motor do teste (a métrica oficial do
                C-MAPSS). Só olhamos no final, para reportar.

Uso (na pasta cmapss-rul, com o .venv ativo):
    python -m cmapss.train_features --data-dir data
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GroupKFold
from xgboost import XGBRegressor

from cmapss.data import last_cycle_rows, load_test, load_train
from cmapss.features import DEFAULT_WINDOWS, add_rolling_features, feature_columns
from cmapss.train_baseline import usable_sensors


def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def make_models(fast: bool = False) -> dict:
    n_rf = 50 if fast else 150
    n_xgb = 100 if fast else 400
    return {
        # max_features<1 e min_samples_leaf>1 deixam o Random Forest bem mais rápido
        # com ~125 variáveis (e ainda regularizam o modelo)
        "RandomForest": lambda: RandomForestRegressor(
            n_estimators=n_rf,
            max_features=0.3,
            min_samples_leaf=5,
            n_jobs=-1,
            random_state=42,
        ),
        "XGBoost": lambda: XGBRegressor(
            n_estimators=n_xgb,
            learning_rate=0.05,
            max_depth=5,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1,
        ),
    }


def cv_rmse(make_model, x: pd.DataFrame, y: pd.Series, groups: pd.Series, n_splits: int = 5) -> float:
    """RMSE médio da validação cruzada agrupada por motor."""
    scores = []
    for tr, va in GroupKFold(n_splits=n_splits).split(x, y, groups):
        model = make_model()
        model.fit(x.iloc[tr], y.iloc[tr])
        scores.append(rmse(y.iloc[va], model.predict(x.iloc[va])))
    return float(np.mean(scores))


def run(
    data_dir: str,
    out_dir: str = "results",
    windows: tuple[int, ...] = DEFAULT_WINDOWS,
    fast: bool = False,
) -> pd.DataFrame:
    train = load_train(data_dir)
    test, rul_true = load_test(data_dir)
    sensors = usable_sensors(train)

    train_f = add_rolling_features(train, sensors, windows)
    test_f = add_rolling_features(test, sensors, windows)
    test_last = last_cycle_rows(test_f)

    rows = []
    for variant, with_rolling in [("sensores brutos", False), ("+ médias móveis", True)]:
        cols = feature_columns(sensors, windows, with_rolling)
        x, y, groups = train_f[cols], train_f["RUL"], train_f["unit"]
        for name, make_model in make_models(fast).items():
            cv = cv_rmse(make_model, x, y, groups)
            model = make_model()
            model.fit(x, y)
            test_rmse = rmse(rul_true, model.predict(test_last[cols]))
            rows.append(
                {
                    "modelo": name,
                    "variaveis": variant,
                    "n_variaveis": len(cols),
                    "rmse_cv": round(cv, 2),
                    "rmse_teste": round(test_rmse, 2),
                }
            )
            print(f"{name:13s} | {variant:16s} | CV {cv:6.2f} | teste {test_rmse:6.2f}", flush=True)

    result = pd.DataFrame(rows)
    Path(out_dir).mkdir(exist_ok=True)
    result.to_csv(Path(out_dir) / "features.csv", index=False)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--fast", action="store_true", help="menos árvores, para testar rápido")
    args = parser.parse_args()
    print()
    print(run(args.data_dir, fast=args.fast).to_string(index=False))
