"""Treina e salva o modelo FINAL (todas as linhas do treino) com a configuração escolhida.

A configuração (variante de variáveis e hiperparâmetros) vem do resultado do ajuste por validação
cruzada, para não digitar nada na mão:

    python -m cmapss.train_final --data-dir data --variant "+norm+ciclo" \\
        --tuning-csv results/tuning_final_norm_ciclo.csv

Saídas:
  models/rul_xgb.json    o modelo (formato nativo do XGBoost)
  models/metadata.json   variáveis, janelas, hiperparâmetros, versão e métricas no teste
  examples/engine_*.txt  históricos de 3 motores do teste (para a demonstração), com a RUL real
  examples/examples.json

As métricas gravadas são do teste (último ciclo), só para descrição; a escolha do modelo foi feita
por validação cruzada.
"""

import argparse
import ast
import json
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor

from cmapss.cap_experiment import BEST_XGB, EVAL_CAP
from cmapss.data import RUL_CAP, last_cycle_rows, load_test, load_train
from cmapss.feature_experiment import nasa_score
from cmapss.features import DEFAULT_WINDOWS
from cmapss.features_v2 import N_BASELINE, SLOPE_WINDOWS, VARIANTS, build_features, feature_columns_v2
from cmapss.predict import META_FILE, MODEL_FILE
from cmapss.train_baseline import usable_sensors

N_EXAMPLES = 3


def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def params_from_tuning(csv_path: str | Path, model_label: str = "XGBoost") -> tuple[dict, str | None]:
    """Lê os hiperparâmetros (e a variante) da tabela `tuning_final*.csv`."""
    df = pd.read_csv(csv_path)
    row = df[df["modelo"] == model_label].iloc[0]
    variant = row["variaveis"] if "variaveis" in df.columns else None
    return ast.literal_eval(row["params"]), variant


def fit_final(
    data_dir: str,
    variant: str = "base",
    params: dict | None = None,
    seed: int = 0,
    windows: tuple[int, ...] = DEFAULT_WINDOWS,
) -> tuple[XGBRegressor, dict, pd.DataFrame, np.ndarray]:
    """Treina com todo o treino. Devolve (modelo, metadados, histórico do teste, RUL real do teste sem cap)."""
    params = BEST_XGB if params is None else params
    train = load_train(data_dir)
    test, rul_raw = load_test(data_dir, cap=None)
    sensors = usable_sensors(train)

    train_f = build_features(train, sensors, windows, SLOPE_WINDOWS, N_BASELINE)
    test_f = build_features(test, sensors, windows, SLOPE_WINDOWS, N_BASELINE)
    cols = feature_columns_v2(sensors, windows, SLOPE_WINDOWS, **VARIANTS[variant])

    model = XGBRegressor(n_jobs=-1, random_state=seed, **params)
    model.fit(train_f[cols], train_f["RUL"])

    pred = model.predict(last_cycle_rows(test_f)[cols])
    truth_125 = np.minimum(rul_raw, EVAL_CAP)
    meta = {
        "variant": variant,
        "features": cols,
        "sensors": sensors,
        "windows": list(windows),
        "slope_windows": list(SLOPE_WINDOWS),
        "n_baseline": N_BASELINE,
        "rul_cap": RUL_CAP,
        "params": params,
        "seed": seed,
        "dataset": "NASA C-MAPSS FD001",
        "xgboost_version": xgboost.__version__,
        "metrics_test": {
            "rmse_real_ate_125": rmse(truth_125, pred),
            "rmse_real_sem_limite": rmse(rul_raw, pred),
            "score_assimetrico": nasa_score(pred - truth_125),
            "pct_erro_positivo": float(np.mean(pred - truth_125 > 0)),
            "n_motores_teste": int(len(rul_raw)),
        },
    }
    return model, meta, test, rul_raw


def save(model: XGBRegressor, meta: dict, out_dir: str | Path) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_model(str(out_dir / MODEL_FILE))
    (out_dir / META_FILE).write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")


def save_examples(test: pd.DataFrame, rul_raw: np.ndarray, examples_dir: str | Path, n: int = N_EXAMPLES) -> None:
    """Salva o histórico de alguns motores do teste, com a RUL real, para a demonstração."""
    examples_dir = Path(examples_dir)
    examples_dir.mkdir(parents=True, exist_ok=True)
    units = sorted(test["unit"].unique())[:n]
    info = {}
    for u in units:
        d = test[test["unit"] == u]
        name = f"engine_{int(u)}.txt"
        d.to_csv(examples_dir / name, sep=" ", header=False, index=False)
        info[name] = {"unit": int(u), "rul_real": int(rul_raw[units.index(u)])}
    (examples_dir / "examples.json").write_text(json.dumps(info, indent=2), encoding="utf-8")


def run(
    data_dir: str,
    out_dir: str = "models",
    examples_dir: str = "examples",
    variant: str = "base",
    params: dict | None = None,
    tuning_csv: str | None = None,
    seed: int = 0,
    windows: tuple[int, ...] = DEFAULT_WINDOWS,
) -> dict:
    if tuning_csv:
        params, tuned_variant = params_from_tuning(tuning_csv)
        if tuned_variant and tuned_variant != variant:
            print(f"Aviso: o CSV foi gerado com a variante '{tuned_variant}', mas você pediu '{variant}'.")
    model, meta, test, rul_raw = fit_final(data_dir, variant, params, seed, windows)
    save(model, meta, out_dir)
    save_examples(test, rul_raw, examples_dir)
    return meta


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--variant", default="base", choices=list(VARIANTS))
    parser.add_argument("--tuning-csv", default=None, help="results/tuning_final*.csv com os hiperparâmetros")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    meta = run(args.data_dir, variant=args.variant, tuning_csv=args.tuning_csv, seed=args.seed)
    m = meta["metrics_test"]
    print(f"Modelo salvo em models/ ({len(meta['features'])} variáveis, variante '{meta['variant']}')")
    print(f"Teste (RUL real <= 125): RMSE {m['rmse_real_ate_125']:.2f} | sem limite: {m['rmse_real_sem_limite']:.2f} "
          f"| score {m['score_assimetrico']:.0f} | erro positivo {m['pct_erro_positivo']:.0%}")
