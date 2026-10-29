import numpy as np
import pandas as pd

from cmapss.data import COLUMNS, RUL_CAP, last_cycle_rows, load_test, load_train
from cmapss.train_baseline import run


def _fake_files(tmp_path, n_units=5, max_cycles=160):
    rng = np.random.default_rng(0)
    rows_train, rows_test, rul = [], [], []
    for unit in range(1, n_units + 1):
        n = int(rng.integers(130, max_cycles))
        for cycle in range(1, n + 1):
            wear = cycle / n
            sensors = rng.normal(0, 0.1, 21) + np.linspace(0, 1, 21) * wear
            sensors[0] = 5.0  # sensor constante de propósito
            rows_train.append([unit, cycle, 0.0, 0.0, 100.0, *sensors])
        n_test = n - 20
        for cycle in range(1, n_test + 1):
            wear = cycle / n
            sensors = rng.normal(0, 0.1, 21) + np.linspace(0, 1, 21) * wear
            sensors[0] = 5.0
            rows_test.append([unit, cycle, 0.0, 0.0, 100.0, *sensors])
        rul.append(20)
    pd.DataFrame(rows_train, columns=COLUMNS).to_csv(tmp_path / "train_FD001.txt", sep=" ", header=False, index=False)
    pd.DataFrame(rows_test, columns=COLUMNS).to_csv(tmp_path / "test_FD001.txt", sep=" ", header=False, index=False)
    pd.Series(rul).to_csv(tmp_path / "RUL_FD001.txt", header=False, index=False)


def test_rul_is_capped_and_ends_at_zero(tmp_path):
    _fake_files(tmp_path)
    train = load_train(tmp_path)
    assert train["RUL"].max() <= RUL_CAP
    assert (train.groupby("unit")["RUL"].min() == 0).all()


def test_last_cycle_one_row_per_unit(tmp_path):
    _fake_files(tmp_path)
    test, rul_true = load_test(tmp_path)
    last = last_cycle_rows(test)
    assert len(last) == test["unit"].nunique() == len(rul_true)


def test_baseline_runs(tmp_path):
    _fake_files(tmp_path)
    result = run(str(tmp_path), out_dir=str(tmp_path / "results"))
    assert set(result["modelo"]) == {"LinearRegression", "RandomForest"}
    assert (result["rmse_teste"] >= 0).all()
