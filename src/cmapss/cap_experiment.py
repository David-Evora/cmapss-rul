"""Experimento: o limite (cap) da RUL no alvo de treino ajuda ou atrapalha?

O cap (125) é uma escolha NOSSA sobre o alvo de treino, não algo que o modelo aprende.
Aqui treinamos o mesmo XGBoost com vários caps e medimos TODOS contra a RUL real sem limite,
em faixas onde a comparação é justa:

  Validação cruzada por motor (nenhum motor em treino e validação ao mesmo tempo):
  - cv_ate75:   linhas com RUL real <= 75. O alvo de treino é idêntico para todos os caps nessa faixa,
                então isso isola: "o cap ajuda a aprender perto da falha?"
  - cv_ate125:  linhas com RUL real <= 125. Mostra também o custo de um cap baixo, que fica "cego"
                entre o cap e 125.

  Teste (último ciclo de cada motor):
  - teste_125:      contra a RUL real limitada a 125 (padrão da literatura).
  - teste_sem_cap:  contra a RUL real SEM limite (a régua mais exigente).

Como ler: se um cap só parece bom em `teste_125` mas piora em `teste_sem_cap`, ele estava
"ajudando" a métrica e não o modelo. Os motores de teste com RUL real acima de 125 são os
que fazem a diferença entre essas duas réguas.

Uso (na pasta cmapss-rul, com o .venv ativo):
    python -m cmapss.cap_experiment --data-dir data
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GroupKFold
from xgboost import XGBRegressor

from cmapss.data import last_cycle_rows, load_test, load_train
from cmapss.features import DEFAULT_WINDOWS, add_rolling_features, feature_columns
from cmapss.train_baseline import usable_sensors

EVAL_CAP = 125  # régua padrão da literatura para o teste

# Melhor configuração do ajuste por validação cruzada (sorteio 18)
BEST_XGB = {
    "subsample": 0.7, "reg_lambda": 1, "n_estimators": 800, "min_child_weight": 3,
    "max_depth": 4, "learning_rate": 0.03, "colsample_bytree": 0.7,
}

DEFAULT_CAPS = (75, 100, 125, 150, None)
METRICS = ["cv_ate75", "cv_ate125", "teste_125", "teste_sem_cap"]


def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def cap_label(cap) -> str:
    return "sem cap" if cap is None else str(cap)


def run(
    data_dir: str,
    out_dir: str = "results",
    caps: tuple = DEFAULT_CAPS,
    seeds: int = 3,
    windows: tuple[int, ...] = DEFAULT_WINDOWS,
    params: dict | None = None,
) -> pd.DataFrame:
    params = BEST_XGB if params is None else params

    train = load_train(data_dir, cap=None)  # RUL e RUL_raw sem limite
    test, rul_raw = load_test(data_dir, cap=None)
    sensors = usable_sensors(train)

    train_f = add_rolling_features(train, sensors, windows)
    test_last = last_cycle_rows(add_rolling_features(test, sensors, windows))
    cols = feature_columns(sensors, windows, with_rolling=True)

    x, groups = train_f[cols], train_f["unit"]
    raw = train_f["RUL_raw"].to_numpy()
    mask75, mask125 = raw <= 75, raw <= 125
    n_above = int((rul_raw > EVAL_CAP).sum())
    print(f"Motores de teste com RUL real > {EVAL_CAP}: {n_above} de {len(rul_raw)}", flush=True)

    rows = []
    for cap in caps:
        y = pd.Series(raw if cap is None else np.minimum(raw, cap), index=x.index)
        for seed in range(seeds):
            oof = np.full(len(x), np.nan)
            for tr, va in GroupKFold(n_splits=5).split(x, y, groups):
                model = XGBRegressor(n_jobs=-1, random_state=seed, **params)
                model.fit(x.iloc[tr], y.iloc[tr])
                oof[va] = model.predict(x.iloc[va])

            final = XGBRegressor(n_jobs=-1, random_state=seed, **params)
            final.fit(x, y)
            pred = final.predict(test_last[cols])

            r = {
                "cap": cap_label(cap),
                "semente": seed,
                "cv_ate75": rmse(raw[mask75], oof[mask75]),
                "cv_ate125": rmse(raw[mask125], oof[mask125]),
                "teste_125": rmse(np.minimum(rul_raw, EVAL_CAP), pred),
                "teste_sem_cap": rmse(rul_raw, pred),
            }
            rows.append(r)
            print(
                f"cap {r['cap']:>8s} | semente {seed} | CV(<=75) {r['cv_ate75']:6.2f} | "
                f"CV(<=125) {r['cv_ate125']:6.2f} | teste(125) {r['teste_125']:6.2f} | "
                f"teste(sem cap) {r['teste_sem_cap']:6.2f}",
                flush=True,
            )

    detail = pd.DataFrame(rows)
    order = [cap_label(c) for c in caps]
    summary = detail.groupby("cap")[METRICS].agg(["mean", "std"]).reindex(order)
    summary.columns = [f"{a}_{'media' if b == 'mean' else 'desvio'}" for a, b in summary.columns]
    summary = summary.reset_index()

    Path(out_dir).mkdir(exist_ok=True)
    detail.round(3).to_csv(Path(out_dir) / "cap_experiment_detalhe.csv", index=False)
    summary.round(3).to_csv(Path(out_dir) / "cap_experiment.csv", index=False)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--seeds", type=int, default=3)
    args = parser.parse_args()

    summary = run(args.data_dir, seeds=args.seeds)
    print("\n=== RESUMO (média ± desvio entre sementes; menor = melhor) ===")
    for _, r in summary.iterrows():
        print(
            f"cap {r['cap']:>8s} | CV(<=75) {r['cv_ate75_media']:.2f} ± {r['cv_ate75_desvio']:.2f} | "
            f"CV(<=125) {r['cv_ate125_media']:.2f} ± {r['cv_ate125_desvio']:.2f} | "
            f"teste(125) {r['teste_125_media']:.2f} ± {r['teste_125_desvio']:.2f} | "
            f"teste(sem cap) {r['teste_sem_cap_media']:.2f} ± {r['teste_sem_cap_desvio']:.2f}"
        )
