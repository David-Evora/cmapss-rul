"""Demonstração em Gradio: prevê a vida útil restante (RUL) de um motor turbofan.

Rodar localmente:   pip install gradio  &&  python app.py
No Hugging Face Spaces este arquivo é o ponto de entrada (veja `python -m cmapss.publish_hf`).
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

import gradio as gr  # noqa: E402

from cmapss.demo import DISCLAIMER, list_examples, predict_from_path  # noqa: E402

MODEL_DIR = ROOT / "models"
EXAMPLES_DIR = ROOT / "examples"


def run(example, upload):
    if upload:
        path = upload
    elif example:
        path = EXAMPLES_DIR / example
    else:
        return "Escolha um exemplo ou envie o histórico de um motor.", None
    try:
        return predict_from_path(path, MODEL_DIR, EXAMPLES_DIR)
    except Exception as exc:  # mostra o motivo em vez de quebrar a página
        return f"Não consegui processar o arquivo: {exc}", None


with gr.Blocks(title="Predição de vida útil de motores (C-MAPSS)") as demo:
    gr.Markdown(
        "# Predição de vida útil restante (RUL) de motores turbofan\n"
        "Modelo XGBoost treinado no NASA C-MAPSS (FD001). Escolha um motor de exemplo do conjunto de teste "
        "ou envie o histórico de sensores de um motor (26 colunas, sem cabeçalho, começando no ciclo 1).\n\n"
        f"_{DISCLAIMER}_"
    )
    with gr.Row():
        example = gr.Dropdown(choices=list_examples(EXAMPLES_DIR), label="Exemplo (motor do conjunto de teste)")
        upload = gr.File(label="Ou envie o histórico (.txt/.csv)", type="filepath")
    button = gr.Button("Prever RUL", variant="primary")
    out_text = gr.Markdown()
    out_plot = gr.Plot()
    button.click(run, inputs=[example, upload], outputs=[out_text, out_plot])

if __name__ == "__main__":
    demo.launch()
