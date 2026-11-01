"""Dia 4b: ajuste de hiperparâmetros guiado por validação cruzada (sem atalhos de velocidade).

Regras que este script segue (para o resultado ser defensável):
  1. A ESCOLHA da configuração usa só o `rmse_cv` (validação cruzada agrupada por motor, no treino).
  2. O conjunto de TESTE só é usado no fim, uma vez por semente, para reportar.
  3. Para diagnosticar underfitting/overfitting, cada configuração também mostra o erro de treino
     (`rmse_treino`) e a diferença para o CV (`gap`):
        - treino e CV ambos altos           -> provável underfitting
        - treino muito menor que o CV (gap alto) -> provável overfitting
  4. As configurações "do Dia 4" e "padrão" entram na disputa, então dá para ver se o atalho
     de velocidade que usamos antes custou ou não precisão.
  5. O melhor modelo é repetido com várias sementes e reportado como média ± desvio.

Uso (na pasta cmapss-rul, com o .venv ativo):
    python -m cmapss.tune --data-dir data                 # tudo (pode demorar bastante)
    python -m cmapss.tune --data-dir data --models xgb    # só XGBoost (bem mais rápido)
"""

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GroupKFold, ParameterSampler
from xgboost import XGBRegressor

from cmapss.data import last_cycle_rows, load_test, load_train
from cmapss.features import DEFAULT_WINDOWS, add_rolling_features, feature_columns
from cmapss.train_baseline import usable_sensors

SEARCH_SEED = 0  # semente do SORTEIO das configurações (não é a semente dos modelos)

RF_SPACE = {
    "n_estimators": [150, 300],
    "max_depth": [None, 10, 20, 30],
    "min_samples_leaf": [1, 3, 5, 10],
    "max_features": [1.0, 0.5, 0.3, "sqrt"],
}
XGB_SPACE = {
    "n_estimators": [200, 400, 600, 800],
    "learning_rate": [0.02, 0.03, 0.05, 0.1],
    "max_depth": [3, 4, 5, 6, 7, 8],
    "min_child_weight": [1, 3, 5, 10],
    "subsample": [0.7, 0.8, 0.9, 1.0],
    "colsample_bytree": [0.5, 0.7, 0.8, 1.0],
    "reg_lambda": [1, 5, 10],
}

# Configurações de referência que sempre entram na disputa
RF_FIXED = {
    "dia4 (max_features=0.3, leaf=5)": {
        "n_estimators": 150, "max_depth": None, "min_samples_leaf": 5, "max_features": 0.3,
    },
    "padrão sklearn (max_features=1.0, leaf=1)": {
        "n_estimators": 150, "max_depth": None, "min_samples_leaf": 1, "max_features": 1.0,
    },
}
XGB_FIXED = {
    "dia4 (depth=5, lr=0.05)": {
        "n_estimators": 400, "learning_rate": 0.05, "max_depth": 5, "min_child_weight": 1,
        "subsample": 0.8, "colsample_bytree": 0.8, "reg_lambda": 1,
    },
}


def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def build(kind: str, params: dict, seed: int):
    if kind == "rf":
        return RandomForestRegressor(n_jobs=-1, random_state=seed, **params)
    return XGBRegressor(n_jobs=-1, random_state=seed, **params)


def evaluate(kind: str, params: dict, x, y, groups, seed: int = 42, n_splits: int = 5) -> dict:
    """Validação cruzada agrupada por motor, medindo erro de treino e de validação."""
    tr_scores, va_scores = [], []
    for tr, va in GroupKFold(n_splits=n_splits).split(x, y, groups):
        model = build(kind, params, seed)
        model.fit(x.iloc[tr], y.iloc[tr])
        tr_scores.append(rmse(y.iloc[tr], model.predict(x.iloc[tr])))
        va_scores.append(rmse(y.iloc[va], model.predict(x.iloc[va])))
    return {
        "rmse_treino": float(np.mean(tr_scores)),
        "rmse_cv": float(np.mean(va_scores)),
        "cv_desvio": float(np.std(va_scores)),
    }


def search(kind: str, space: dict, fixed: dict, n_iter: int, x, y, groups) -> pd.DataFrame:
    """Avalia as configurações fixas e `n_iter` configurações sorteadas; ordena pelo CV."""
    candidates = [(name, p) for name, p in fixed.items()]
    sampled = list(ParameterSampler(space, n_iter=n_iter, random_state=SEARCH_SEED))
    candidates += [(f"sorteio {i + 1}", p) for i, p in enumerate(sampled)]

    rows = []
    t_start = time.time()
    for i, (name, params) in enumerate(candidates, 1):
        t0 = time.time()
        res = evaluate(kind, params, x, y, groups)
        rows.append({"config": name, **res, "gap": res["rmse_cv"] - res["rmse_treino"],
                     "segundos": round(time.time() - t0, 1), "params": params})
        elapsed = time.time() - t_start
        eta = elapsed / i * (len(candidates) - i)
        print(
            f"[{kind}] {i:2d}/{len(candidates)} {name:42s} "
            f"treino {res['rmse_treino']:6.2f} | CV {res['rmse_cv']:6.2f} "
            f"(±{res['cv_desvio']:.2f}) | faltam ~{eta / 60:.1f} min",
            flush=True,
        )
    return pd.DataFrame(rows).sort_values("rmse_cv").reset_index(drop=True)


def final_report(kind: str, params: dict, x, y, groups, test_x, rul_true, seeds: int) -> dict:
    """Repete o melhor modelo com várias sementes. Aqui (e só aqui) o teste é usado."""
    cvs, tests = [], []
    for seed in range(seeds):
        cvs.append(evaluate(kind, params, x, y, groups, seed=seed)["rmse_cv"])
        model = build(kind, params, seed)
        model.fit(x, y)
        tests.append(rmse(rul_true, model.predict(test_x)))
        print(f"[{kind}] semente {seed}: CV {cvs[-1]:.2f} | teste {tests[-1]:.2f}", flush=True)
    return {
        "rmse_cv_media": float(np.mean(cvs)), "rmse_cv_desvio": float(np.std(cvs)),
        "rmse_teste_media": float(np.mean(tests)), "rmse_teste_desvio": float(np.std(tests)),
    }


def run(
    data_dir: str,
    out_dir: str = "results",
    models: tuple[str, ...] = ("rf", "xgb"),
    rf_iter: int = 8,
    xgb_iter: int = 30,
    seeds: int = 5,
    windows: tuple[int, ...] = DEFAULT_WINDOWS,
) -> pd.DataFrame:
    train = load_train(data_dir)
    test, rul_true = load_test(data_dir)
    sensors = usable_sensors(train)

    train_f = add_rolling_features(train, sensors, windows)
    test_last = last_cycle_rows(add_rolling_features(test, sensors, windows))
    cols = feature_columns(sensors, windows, with_rolling=True)
    x, y, groups = train_f[cols], train_f["RUL"], train_f["unit"]

    plan = {
        "rf": (RF_SPACE, RF_FIXED, rf_iter, "Random Forest"),
        "xgb": (XGB_SPACE, XGB_FIXED, xgb_iter, "XGBoost"),
    }
    Path(out_dir).mkdir(exist_ok=True)
    final_rows = []
    for kind in models:
        space, fixed, n_iter, label = plan[kind]
        table = search(kind, space, fixed, n_iter, x, y, groups)
        table.drop(columns=["params"]).round(3).to_csv(Path(out_dir) / f"tuning_{kind}.csv", index=False)
        print(f"\n=== {label}: configurações ordenadas pelo CV ===")
        print(table.drop(columns=["params"]).round(2).to_string(index=False))

        best = table.iloc[0]
        print(f"\nMelhor ({label}): {best['config']} -> {best['params']}\n")
        rep = final_report(kind, best["params"], x, y, groups, test_last[cols], rul_true, seeds)
        final_rows.append({"modelo": label, "config": best["config"], **rep, "params": str(best["params"])})

    final = pd.DataFrame(final_rows)
    final.round(3).to_csv(Path(out_dir) / "tuning_final.csv", index=False)
    return final


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--models", default="rf,xgb", help="rf, xgb ou rf,xgb")
    parser.add_argument("--rf-iter", type=int, default=8, help="nº de configurações sorteadas do RF")
    parser.add_argument("--xgb-iter", type=int, default=30, help="nº de configurações sorteadas do XGBoost")
    parser.add_argument("--seeds", type=int, default=5)
    args = parser.parse_args()

    result = run(
        args.data_dir,
        models=tuple(m.strip() for m in args.models.split(",")),
        rf_iter=args.rf_iter,
        xgb_iter=args.xgb_iter,
        seeds=args.seeds,
    )
    print("\n=== RESULTADO FINAL (melhor de cada modelo, média ± desvio entre sementes) ===")
    for _, r in result.iterrows():
        print(
            f"{r['modelo']:14s} CV {r['rmse_cv_media']:.2f} ± {r['rmse_cv_desvio']:.2f} | "
            f"teste {r['rmse_teste_media']:.2f} ± {r['rmse_teste_desvio']:.2f} | {r['config']}"
        )
