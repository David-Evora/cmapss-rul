import numpy as np

from cmapss.cap_experiment import run
from cmapss.data import RUL_CAP, load_test, load_train

from test_data import _fake_files


def test_cap_parameter_in_loaders(tmp_path):
    _fake_files(tmp_path, n_units=5)

    capped = load_train(tmp_path)  # padrão: cap de 125
    raw = load_train(tmp_path, cap=None)
    assert capped["RUL"].max() <= RUL_CAP
    assert raw["RUL"].max() > RUL_CAP  # os motores de teste tem 130+ ciclos
    assert (capped["RUL_raw"] == raw["RUL_raw"]).all()  # RUL_raw nunca é limitada

    small = load_train(tmp_path, cap=50)
    assert small["RUL"].max() <= 50

    _, rul_default = load_test(tmp_path)
    _, rul_free = load_test(tmp_path, cap=None)
    assert rul_default.max() <= RUL_CAP
    assert np.array_equal(rul_free, np.array([20] * 5))


def test_cap_experiment_runs(tmp_path):
    _fake_files(tmp_path, n_units=8)
    out = tmp_path / "res"
    summary = run(str(tmp_path), out_dir=str(out), caps=(50, None), seeds=1, windows=(5,),
                  params={"n_estimators": 20, "learning_rate": 0.1, "max_depth": 3})
    assert list(summary["cap"]) == ["50", "sem cap"]
    for col in ("cv_ate75_media", "cv_ate125_media", "teste_125_media", "teste_sem_cap_media"):
        assert (summary[col] >= 0).all()
    assert (out / "cap_experiment.csv").exists()
