import pandas as pd

from cmapss.tune import RF_FIXED, XGB_FIXED, evaluate, run

from test_data import _fake_files


def test_fixed_configs_are_the_ones_we_used_before():
    assert RF_FIXED["dia4 (max_features=0.3, leaf=5)"]["max_features"] == 0.3
    assert XGB_FIXED["dia4 (depth=5, lr=0.05)"]["max_depth"] == 5


def test_evaluate_reports_train_and_cv(tmp_path):
    from cmapss.data import load_train
    from cmapss.features import add_rolling_features, feature_columns
    from cmapss.train_baseline import usable_sensors

    _fake_files(tmp_path, n_units=8)
    train = load_train(tmp_path)
    sensors = usable_sensors(train)
    tf = add_rolling_features(train, sensors, (5,))
    cols = feature_columns(sensors, (5,))
    res = evaluate(
        "xgb",
        {"n_estimators": 20, "learning_rate": 0.1, "max_depth": 3},
        tf[cols], tf["RUL"], tf["unit"],
    )
    assert set(res) == {"rmse_treino", "rmse_cv", "cv_desvio"}
    assert res["rmse_treino"] <= res["rmse_cv"] + 1e-9  # treino nunca é pior que validação aqui


def test_run_end_to_end_small(tmp_path):
    _fake_files(tmp_path, n_units=8)
    out = tmp_path / "res"
    final = run(str(tmp_path), out_dir=str(out), rf_iter=1, xgb_iter=1, seeds=2, windows=(5,))
    assert set(final["modelo"]) == {"Random Forest", "XGBoost"}
    assert (out / "tuning_rf.csv").exists() and (out / "tuning_xgb.csv").exists()
    table = pd.read_csv(out / "tuning_xgb.csv")
    assert list(table["rmse_cv"]) == sorted(table["rmse_cv"])  # ordenado pelo CV
    assert (final["rmse_teste_desvio"] >= 0).all()
