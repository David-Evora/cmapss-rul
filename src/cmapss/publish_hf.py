"""Prepara (e publica) o modelo e a demonstração no Hugging Face.

Primeiro gere o modelo final:   python -m cmapss.train_final ...
Depois, para só montar as pastas e conferir antes de publicar:

    python -m cmapss.publish_hf --user David-Evora --dry-run

Para publicar de verdade (precisa de uma conta e de um token de ESCRITA do Hugging Face;
faça o login uma vez com:  python -c "from huggingface_hub import login; login()" ):

    pip install huggingface_hub gradio
    python -m cmapss.publish_hf --user David-Evora

Cria dois repositórios: o MODELO (`<user>/cmapss-rul-xgboost`) e a DEMO em Spaces (`<user>/cmapss-rul-demo`).
"""

import argparse
import json
import shutil
from pathlib import Path

from cmapss.predict import META_FILE, MODEL_FILE

APP_MODULES = ["__init__.py", "data.py", "features.py", "features_v2.py", "predict.py", "demo.py"]
SPACE_REQUIREMENTS = "gradio\nxgboost\nscikit-learn\npandas\nnumpy\nmatplotlib\n"


def model_card(meta: dict, repo_url: str, license_id: str) -> str:
    m = meta["metrics_test"]
    return f"""---
license: {license_id}
library_name: xgboost
pipeline_tag: tabular-regression
tags:
- predictive-maintenance
- remaining-useful-life
- turbofan
- c-mapss
- xgboost
---

# C-MAPSS RUL predictor (XGBoost)

Predicts the **Remaining Useful Life (RUL)** of turbofan engines from their sensor history, using the
NASA C-MAPSS dataset (subset FD001, simulated data). Educational / portfolio project.

- Code, experiments and full write-up: {repo_url}
- Live demo: see the companion Space `cmapss-rul-demo`.

## Model

- Algorithm: XGBoost regressor, {len(meta["features"])} input features (feature set `{meta["variant"]}`).
- Features (all **causal**: each cycle only uses the past of the same engine): raw sensors, rolling mean/std
  over {", ".join(str(w) for w in meta["windows"])} cycles; depending on the feature set, also deviation from the
  engine's initial level, sensor slopes and the cycle number.
- Target: RUL **capped at {meta["rul_cap"]} cycles** (a common choice for C-MAPSS: while an engine is healthy
  the sensors carry no information about how long it will last). The model therefore never predicts above {meta["rul_cap"]}.
- Hyperparameters: `{json.dumps(meta["params"])}`, selected by 5-fold cross-validation grouped by engine
  (no engine appears in training and validation at the same time). The test set was used only for reporting.

## Evaluation (FD001 test set, last observed cycle of each of the {m["n_motores_teste"]} test engines)

| Metric | Value |
|---|---|
| RMSE vs RUL capped at 125 (usual protocol) | {m["rmse_real_ate_125"]:.2f} |
| RMSE vs uncapped RUL (stricter) | {m["rmse_real_sem_limite"]:.2f} |
| Asymmetric C-MAPSS score (lower is better) | {m["score_assimetrico"]:.0f} |
| Share of engines with positive error (RUL overestimated) | {m["pct_erro_positivo"]:.0%} |

Cross-validation results, the feature ablation, the cap sensitivity study and the error analysis are in the repository.

## Intended use and limitations

- Demonstration on **simulated** data (single operating condition, FD001); not validated for real engines.
- The history must start at cycle 1, because the features depend on the engine's early cycles.
- Overestimating the remaining life is the costlier mistake in maintenance; see the error analysis in the repository
  for where the model tends to do so.

## Usage

```bash
git clone {repo_url}
cd cmapss-rul && pip install -e .
```

```python
from cmapss.predict import load_model, read_history, predict_last_cycle

model, meta = load_model("models")            # folder with {MODEL_FILE} and {META_FILE}
history = read_history("examples/engine_1.txt")
print(predict_last_cycle(history, model, meta))
```
"""


def space_readme() -> str:
    return """---
title: C-MAPSS RUL Predictor
emoji: 🔧
colorFrom: blue
colorTo: indigo
sdk: gradio
app_file: app.py
pinned: false
---

Demo: remaining useful life (RUL) prediction for turbofan engines (NASA C-MAPSS, FD001).
Educational project, trained on simulated data.
"""


def stage(
    model_dir: str | Path,
    examples_dir: str | Path,
    stage_dir: str | Path,
    repo_url: str,
    license_id: str = "mit",
    repo_root: str | Path | None = None,
) -> tuple[Path, Path]:
    """Monta as duas pastas (modelo e Space) prontas para subir."""
    repo_root = Path(repo_root) if repo_root else Path(__file__).resolve().parents[2]
    model_dir, examples_dir, stage_dir = Path(model_dir), Path(examples_dir), Path(stage_dir)
    meta = json.loads((model_dir / META_FILE).read_text(encoding="utf-8"))

    if stage_dir.exists():
        shutil.rmtree(stage_dir)
    model_stage, space_stage = stage_dir / "model", stage_dir / "space"

    # --- repositório do MODELO ---
    model_stage.mkdir(parents=True)
    shutil.copy(model_dir / MODEL_FILE, model_stage / MODEL_FILE)
    shutil.copy(model_dir / META_FILE, model_stage / META_FILE)
    (model_stage / "README.md").write_text(model_card(meta, repo_url, license_id), encoding="utf-8")

    # --- repositório do SPACE (demo) ---
    (space_stage / "src" / "cmapss").mkdir(parents=True)
    for name in APP_MODULES:
        shutil.copy(repo_root / "src" / "cmapss" / name, space_stage / "src" / "cmapss" / name)
    shutil.copy(repo_root / "app.py", space_stage / "app.py")
    (space_stage / "models").mkdir()
    shutil.copy(model_dir / MODEL_FILE, space_stage / "models" / MODEL_FILE)
    shutil.copy(model_dir / META_FILE, space_stage / "models" / META_FILE)
    if examples_dir.exists():
        shutil.copytree(examples_dir, space_stage / "examples")
    (space_stage / "requirements.txt").write_text(SPACE_REQUIREMENTS, encoding="utf-8")
    (space_stage / "README.md").write_text(space_readme(), encoding="utf-8")
    return model_stage, space_stage


def upload(model_stage: Path, space_stage: Path, user: str, model_name: str, space_name: str) -> None:
    try:
        from huggingface_hub import HfApi
    except ImportError as exc:
        raise SystemExit("Instale o cliente do Hugging Face:  pip install huggingface_hub") from exc

    api = HfApi()  # usa o token do login salvo ou da variável de ambiente HF_TOKEN
    model_id, space_id = f"{user}/{model_name}", f"{user}/{space_name}"

    api.create_repo(repo_id=model_id, repo_type="model", exist_ok=True)
    api.upload_folder(folder_path=str(model_stage), repo_id=model_id, repo_type="model")
    print(f"Modelo publicado: https://huggingface.co/{model_id}")

    api.create_repo(repo_id=space_id, repo_type="space", space_sdk="gradio", exist_ok=True)
    api.upload_folder(folder_path=str(space_stage), repo_id=space_id, repo_type="space")
    print(f"Demo publicada:   https://huggingface.co/spaces/{space_id}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--user", required=True, help="seu usuário no Hugging Face")
    parser.add_argument("--model-dir", default="models")
    parser.add_argument("--examples-dir", default="examples")
    parser.add_argument("--stage-dir", default="hf_stage")
    parser.add_argument("--repo-url", default="https://github.com/David-Evora/cmapss-rul")
    parser.add_argument("--license", default="mit", help="licença do modelo (confirme a sua escolha)")
    parser.add_argument("--model-name", default="cmapss-rul-xgboost")
    parser.add_argument("--space-name", default="cmapss-rul-demo")
    parser.add_argument("--dry-run", action="store_true", help="só monta as pastas, não publica")
    args = parser.parse_args()

    model_stage, space_stage = stage(
        args.model_dir, args.examples_dir, args.stage_dir, args.repo_url, args.license
    )
    print(f"Pastas montadas em {args.stage_dir}/ (model/ e space/). Confira o conteúdo antes de publicar.")
    if not args.dry_run:
        upload(model_stage, space_stage, args.user, args.model_name, args.space_name)
