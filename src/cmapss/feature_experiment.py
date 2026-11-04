"""Experimento: as variáveis novas reduzem o atraso/viés do modelo?

Compara, com o MESMO XGBoost ajustado e a MESMA validação (agrupada por motor), cinco conjuntos
de variáveis:

  base                  sensores + médias/desvios móveis (o que já tínhamos)
  +normalização         + desvio em relação ao estado inicial de cada motor
  +inclinação           + inclinação (tendência) dos sensores
  +ciclo                + o número do ciclo
  (e as combinações: +norm+incl, +norm+ciclo, +incl+ciclo, +norm+incl+ciclo)

Métricas (menor = melhor, exceto onde dito):
  cv_125         RMSE da validação cruzada contra o alvo limitado a 125 (todas as linhas)
  cv_31_100      RMSE só nas linhas com RUL real de 31 a 100, a zona de transição onde mais erramos
  vies_31_100    erro médio nessa zona (positivo = otimista; quanto mais perto de 0, melhor)
  corr_vida_vies correlação, por motor, entre a vida do motor e o viés médio (linhas com RUL <= 100);
                 o diagnóstico anterior deu ~ -0,65 (regressão à média); perto de 0 é melhor
  teste_125      RMSE no último ciclo do teste (RUL real limitada a 125)
  teste_sem_cap  RMSE no teste contra a RUL real sem limite
  score_teste    score assimétrico do C-MAPSS no teste (pune mais superestimar a vida)

A decisão sobre qual conjunto manter usa as métricas de VALIDAÇÃO (cv_*). O teste só descreve.
Compare as linhas ENTRE SI (o cv_125 aqui é um RMSE agrupado, não idêntico ao do ajuste anterior).

Uso (na pasta cmapss-rul, com o .venv ativo):
    python -m cmapss.feature_experiment --data-dir data
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GroupKFold
from xgboost import XGBRegressor

from cmapss.cap_experiment import BEST_XGB, EVAL_CAP
from cmapss.data import last_cycle_rows, load_test, load_train
from cmapss.features import DEFAULT_WINDOWS
from cmapss.features_v2 import N_BASELINE, SLOPE_WINDOWS, VARIANTS, build_features, feature_columns_v2
from cmapss.train_baseline import usable_sensors

METRICS = [
    "cv_125", "cv_31_100", "vies_31_100", "corr_vida_vies",
    "teste_125", "teste_sem_cap", "score_teste",
]


def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def nasa_score(d: np.ndarray) -> float:
    """Score assimétrico do desafio C-MAPSS (d = previsto - real); pune mais d > 0."""
    return float(np.sum(np.where(d < 0, np.exp(-d / 13) - 1, np.exp(d / 10) - 1)))


def run(
    data_dir: str,
    out_dir: str = "results",
    seeds: int = 3,
    windows: tuple[int, ...] = DEFAULT_WINDOWS,
    slope_windows: tuple[int, ...] = SLOPE_WINDOWS,
    n_baseline: int = N_BASELINE,
    params: dict | None = None,
    variants: dict | None = None,
    out_name: str = "feature_experiment",
) -> pd.DataFrame:
    params = BEST_XGB if params is None else params
    variants = VARIANTS if variants is None else variants

    train = load_train(data_dir)  # alvo RUL limitado a 125; RUL_raw sem limite
    test, rul_raw = load_test(data_dir, cap=None)
    sensors = usable_sensors(train)

    train_f = build_features(train, sensors, windows, slope_windows, n_baseline)
    test_last = last_cycle_rows(build_features(test, sensors, windows, slope_windows, n_baseline))

    y = train_f["RUL"]
    raw = train_f["RUL_raw"].to_numpy()
    groups = train_f["unit"]
    vida = train_f.groupby("unit")["cycle"].transform("max").to_numpy()
    zone = (raw >= 31) & (raw <= 100)
    low = raw <= 100
    truth_test = np.minimum(rul_raw, EVAL_CAP)

    rows = []
    for name, flags in variants.items():
        cols = feature_columns_v2(sensors, windows, slope_windows, **flags)
        x = train_f[cols]
        for seed in range(seeds):
            oof = np.full(len(x), np.nan)
            for tr, va in GroupKFold(n_splits=5).split(x, y, groups):
                model = XGBRegressor(n_jobs=-1, random_state=seed, **params)
                model.fit(x.iloc[tr], y.iloc[tr])
                oof[va] = model.predict(x.iloc[va])

            err = oof - raw
            per_unit = (
                pd.DataFrame({"unit": groups.to_numpy()[low], "err": err[low], "vida": vida[low]})
                .groupby("unit")
                .agg(vies=("err", "mean"), vida=("vida", "first"))
            )

            final = XGBRegressor(n_jobs=-1, random_state=seed, **params)
            final.fit(x, y)
            pred = final.predict(test_last[cols])

            r = {
                "variaveis": name,
                "n_variaveis": len(cols),
                "semente": seed,
                "cv_125": rmse(y.to_numpy(), oof),
                "cv_31_100": rmse(raw[zone], oof[zone]),
                "vies_31_100": float(err[zone].mean()),
                "corr_vida_vies": float(per_unit["vida"].corr(per_unit["vies"])),
                "teste_125": rmse(truth_test, pred),
                "teste_sem_cap": rmse(rul_raw, pred),
                "score_teste": nasa_score(pred - truth_test),
            }
            rows.append(r)
            print(
                f"{name:18s} | sem {seed} | CV {r['cv_125']:5.2f} | zona 31-100 {r['cv_31_100']:5.2f} "
                f"(viés {r['vies_31_100']:+5.2f}) | corr {r['corr_vida_vies']:+.2f} | "
                f"teste {r['teste_125']:5.2f} | score {r['score_teste']:7.1f}",
                flush=True,
            )

    detail = pd.DataFrame(rows)
    order = list(variants)
    summary = detail.groupby("variaveis")[METRICS].agg(["mean", "std"]).reindex(order)
    summary.columns = [f"{a}_{'media' if b == 'mean' else 'desvio'}" for a, b in summary.columns]
    n_feat = detail.groupby("variaveis")["n_variaveis"].first().reindex(order)
    summary.insert(0, "n_variaveis", n_feat)
    summary = summary.reset_index()

    Path(out_dir).mkdir(exist_ok=True)
    detail.round(3).to_csv(Path(out_dir) / f"{out_name}_detalhe.csv", index=False)
    summary.round(3).to_csv(Path(out_dir) / f"{out_name}.csv", index=False)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument(
        "--only",
        default=None,
        help='nomes separados por vírgula (ex.: "+ciclo,+norm+ciclo,+incl+ciclo"); '
        "grava em results/feature_experiment_extra.csv para não sobrescrever o resultado anterior",
    )
    args = parser.parse_args()

    if args.only:
        names = [n.strip() for n in args.only.split(",")]
        unknown = [n for n in names if n not in VARIANTS]
        if unknown:
            raise SystemExit(f"Conjuntos desconhecidos: {unknown}. Opções: {list(VARIANTS)}")
        summary = run(
            args.data_dir, seeds=args.seeds,
            variants={n: VARIANTS[n] for n in names}, out_name="feature_experiment_extra",
        )
    else:
        summary = run(args.data_dir, seeds=args.seeds)
    print("\n=== RESUMO (média ± desvio entre sementes) ===")
    for _, r in summary.iterrows():
        print(
            f"{r['variaveis']:18s} ({int(r['n_variaveis']):3d} var) | "
            f"CV {r['cv_125_media']:.2f} ± {r['cv_125_desvio']:.2f} | "
            f"zona 31-100 {r['cv_31_100_media']:.2f} ± {r['cv_31_100_desvio']:.2f} "
            f"(viés {r['vies_31_100_media']:+.2f}) | corr {r['corr_vida_vies_media']:+.2f} | "
            f"teste {r['teste_125_media']:.2f} ± {r['teste_125_desvio']:.2f} | "
            f"sem cap {r['teste_sem_cap_media']:.2f} | score {r['score_teste_media']:.0f}"
        )
