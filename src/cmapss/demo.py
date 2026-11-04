"""Lógica da demonstração (sem depender do Gradio, para poder ser testada)."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from cmapss.predict import load_model, predict_trajectory, read_history  # noqa: E402

DISCLAIMER = (
    "Demonstração educacional treinada no NASA C-MAPSS (FD001, dados simulados). "
    "Não deve ser usada em motores reais. O histórico precisa começar no ciclo 1."
)


def list_examples(examples_dir: str | Path = "examples") -> list[str]:
    path = Path(examples_dir) / "examples.json"
    if not path.exists():
        return []
    return sorted(json.loads(path.read_text(encoding="utf-8")).keys())


def _true_rul(file_name: str, examples_dir: str | Path) -> int | None:
    path = Path(examples_dir) / "examples.json"
    if not path.exists():
        return None
    info = json.loads(path.read_text(encoding="utf-8")).get(file_name)
    return None if info is None else int(info["rul_real"])


def predict_from_path(
    path: str | Path, model_dir: str | Path = "models", examples_dir: str | Path = "examples"
):
    """Devolve (texto em markdown, figura matplotlib) com a RUL prevista ao longo do histórico."""
    model, meta = load_model(model_dir)
    df = read_history(path)

    note = ""
    if df["unit"].nunique() > 1:
        first = df["unit"].iloc[0]
        df = df[df["unit"] == first]
        note = f"\n\nO arquivo tem vários motores; usei só o motor {int(first)}."

    traj = predict_trajectory(df, model, meta).sort_values("cycle")
    pred = np.maximum(traj["rul_prevista"].to_numpy(), 0.0)  # RUL negativa não faz sentido
    cycles = traj["cycle"].to_numpy()
    last = float(pred[-1])
    cap = meta.get("rul_cap", 125)

    fig, ax = plt.subplots(figsize=(7, 3.6))
    ax.plot(cycles, pred, lw=1.6, label="RUL prevista")
    ax.axhline(cap, color="gray", ls=":", lw=1, label=f"teto do treino ({cap})")
    truth = _true_rul(Path(path).name, examples_dir)
    if truth is not None:
        ax.scatter([cycles[-1]], [min(truth, cap)], color="crimson", zorder=5,
                   label=f"RUL real no último ciclo ({truth})")
    ax.set_xlabel("ciclo")
    ax.set_ylabel("RUL (ciclos restantes)")
    ax.set_title("RUL prevista ao longo do histórico do motor")
    ax.grid(alpha=0.3)
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()

    text = (
        f"**Ciclos observados:** {len(cycles)}  \n"
        f"**RUL prevista no último ciclo:** {last:.0f} ciclos"
    )
    if last >= cap - 5:
        text += f"  \n(perto do teto de {cap}: o motor aparenta estar saudável; o modelo não prevê acima disso)"
    if truth is not None:
        text += f"  \n**RUL real (exemplo do conjunto de teste):** {truth} ciclos"
    text += note + f"\n\n_{DISCLAIMER}_"
    return text, fig
