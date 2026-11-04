import json

import numpy as np
import pandas as pd

from cmapss.data import load_test
from cmapss.demo import list_examples, predict_from_path
from cmapss.predict import load_model, predict_last_cycle, predict_trajectory, read_history
from cmapss.publish_hf import stage
from cmapss.train_final import fit_final, params_from_tuning, run, save

from test_data import _fake_files

SMALL = {"n_estimators": 20, "learning_rate": 0.1, "max_depth": 3}


def test_save_and_load_roundtrip(tmp_path):
    _fake_files(tmp_path, n_units=8)
    model, meta, test, _ = fit_final(str(tmp_path), "+norm+incl+ciclo", SMALL, windows=(5,))
    save(model, meta, tmp_path / "models")
    loaded, meta2 = load_model(tmp_path / "models")
    assert meta2["features"] == meta["features"]

    a = predict_last_cycle(test, model, meta)["rul_prevista"].to_numpy()
    b = predict_last_cycle(test, loaded, meta2)["rul_prevista"].to_numpy()
    assert np.allclose(a, b, atol=1e-4)  # o modelo salvo prevê o mesmo que o da memória
    assert len(a) == test["unit"].nunique()
    assert np.isfinite(a).all()


def test_prediction_is_causal(tmp_path):
    """A previsão de um ciclo não pode depender dos ciclos seguintes."""
    _fake_files(tmp_path, n_units=8)
    model, meta, test, _ = fit_final(str(tmp_path), "+norm+incl+ciclo", SMALL, windows=(5,))
    one = test[test["unit"] == 1]
    full = predict_trajectory(one, model, meta)["rul_prevista"].to_numpy()
    cut = predict_trajectory(one.iloc[:60], model, meta)["rul_prevista"].to_numpy()
    assert np.allclose(full[:60], cut, atol=1e-4)


def test_run_writes_files_and_examples(tmp_path):
    _fake_files(tmp_path, n_units=8)
    meta = run(str(tmp_path), out_dir=str(tmp_path / "models"), examples_dir=str(tmp_path / "examples"),
               variant="+norm+ciclo", params=SMALL, windows=(5,))
    assert (tmp_path / "models" / "rul_xgb.json").exists()
    assert (tmp_path / "models" / "metadata.json").exists()
    assert meta["metrics_test"]["n_motores_teste"] == 8
    names = list_examples(tmp_path / "examples")
    assert len(names) == 3
    hist = read_history(tmp_path / "examples" / names[0])
    assert hist.shape[1] == 26


def test_params_from_tuning_csv(tmp_path):
    csv = tmp_path / "t.csv"
    pd.DataFrame([{"modelo": "XGBoost", "variaveis": "+norm+ciclo", "params": str(SMALL)}]).to_csv(csv, index=False)
    params, variant = params_from_tuning(csv)
    assert params == SMALL and variant == "+norm+ciclo"


def test_read_history_tolerates_trailing_spaces(tmp_path):
    line = " ".join(["1", "1", "0.1", "0.2", "100"] + ["1.5"] * 21) + " \n"
    p = tmp_path / "h.txt"
    p.write_text(line * 3)
    assert read_history(p).shape == (3, 26)


def test_demo_returns_text_and_figure(tmp_path):
    _fake_files(tmp_path, n_units=8)
    run(str(tmp_path), out_dir=str(tmp_path / "models"), examples_dir=str(tmp_path / "examples"),
        variant="+norm+ciclo", params=SMALL, windows=(5,))
    name = list_examples(tmp_path / "examples")[0]
    text, fig = predict_from_path(tmp_path / "examples" / name, tmp_path / "models", tmp_path / "examples")
    assert "RUL prevista" in text and "RUL real" in text
    assert hasattr(fig, "savefig")


def test_publish_stage_builds_expected_tree(tmp_path):
    _fake_files(tmp_path, n_units=8)
    run(str(tmp_path), out_dir=str(tmp_path / "models"), examples_dir=str(tmp_path / "examples"),
        variant="base", params=SMALL, windows=(5,))
    model_stage, space_stage = stage(
        tmp_path / "models", tmp_path / "examples", tmp_path / "hf_stage", "https://github.com/x/y"
    )
    card = (model_stage / "README.md").read_text(encoding="utf-8")
    assert card.startswith("---") and "pipeline_tag: tabular-regression" in card
    assert "https://github.com/x/y" in card
    assert (model_stage / "rul_xgb.json").exists()
    for rel in ("app.py", "requirements.txt", "README.md", "models/rul_xgb.json",
                "src/cmapss/predict.py", "src/cmapss/features_v2.py", "src/cmapss/demo.py"):
        assert (space_stage / rel).exists(), rel
    assert "sdk: gradio" in (space_stage / "README.md").read_text(encoding="utf-8")
    meta = json.loads((model_stage / "metadata.json").read_text(encoding="utf-8"))
    assert meta["variant"] == "base"
    # a demo só carrega arquivos que foram copiados (sem depender de módulos de treino)
    src = (space_stage / "src" / "cmapss" / "predict.py").read_text(encoding="utf-8")
    assert "cap_experiment" not in src
