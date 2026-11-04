# Predição de vida útil de motores (NASA C-MAPSS)

Modelo de machine learning que prevê quantos ciclos de operação ainda restam (RUL, *Remaining Useful Life*) de um motor turbofan a partir das leituras dos sensores, para apoiar **manutenção preditiva**. Inspirado no meu projeto de monitoramento e manutenção preditiva do Grand Prix SENAI (2º lugar na América Latina).

**Demo e modelo:** em breve no Hugging Face (links aqui depois da publicação).

## Dados
NASA C-MAPSS, subconjunto FD001: 100 motores de treino (rodam até a falha), 100 de teste (cortados antes da falha, com a RUL real em arquivo separado), 21 sensores e 3 condições operacionais. Dados **simulados**, com uma única condição de operação.

## O que foi feito

| Etapa | O que foi implementado |
|---|---|
| Exploração | Sensores constantes removidos, correlações e trajetórias dos sensores (notebook 01) |
| Análise do alvo | RUL limitada a 125 ciclos (motor saudável não dá sinal de quando vai falhar); sensibilidade testada com tetos de 75 a sem limite |
| Variáveis | Médias e desvios móveis, desvio em relação ao nível inicial do motor, inclinações e número do ciclo. Todas **causais** (só usam o passado do mesmo motor) |
| Modelagem | Regressão Linear, Random Forest e XGBoost |
| Validação | Validação cruzada em 5 partes **agrupada por motor** (nenhum motor aparece no treino e na validação); várias sementes; teste usado só para relatar |
| Otimização | Busca aleatória de hiperparâmetros por validação cruzada |
| Ablação | 8 combinações de variáveis comparadas (desvio inicial, inclinação, ciclo) |
| Análise de erro | Erro por fase da vida do motor, viés e regressão à média (notebooks 02 e 03) |
| Entrega | Modelo salvo, demo em Gradio e publicação no Hugging Face (`app.py`, `src/cmapss/publish_hf.py`) |

Não se aplicam a este conjunto: imputação de valores ausentes, codificação de categorias e remoção de outliers (os dados são numéricos, completos e simulados).

## Resultados (último ciclo de cada um dos 100 motores de teste)

| Modelo | RMSE (real ≤125) |
|---|---|
| Regressão Linear (baseline) | 20,82 |
| Random Forest (baseline) | 17,16 |
| XGBoost ajustado, variáveis base | ≈16 |
| **XGBoost final (`+norm+ciclo`)** | **≈11** (10,98 no modelo publicado; 11,15 ± 0,10 na média de sementes) |

Modelo final contra o XGBoost base (mesmos hiperparâmetros, seed 0):

| | RMSE (≤125) | RMSE (sem teto) | Score assimétrico | % erro positivo |
|---|---|---|---|---|
| Base | 16,16 | 17,39 | 645,9 | 59% |
| Final | 10,98 | 12,12 | 171,6 | 50% |

Na validação cruzada o padrão é o mesmo (≈15,3 → ≈9,7).

**O que funcionou:** a normalização pelo nível inicial do motor e o número do ciclo. O ganho veio principalmente na faixa de 31 a 100 ciclos restantes (por exemplo, RMSE de 15,85 para 5,63 em 31–60 e de 23,09 para 13,77 em 61–100).

**O que não funcionou / limitações:**
- A regressão à média continua: motores de vida curta são superestimados e os de vida longa subestimados (correlação vida × erro de −0,50 no base e −0,54 no final, com n = 100 testes).
- 50% dos motores ainda têm a RUL superestimada, que é o erro mais caro em manutenção.
- Na faixa de mais de 100 ciclos restantes o erro quase não mudou.
- Dados simulados, uma única condição de operação (FD001) e alvo limitado a 125 ciclos: não é validação para motores reais.
- A escolha das variáveis e dos hiperparâmetros foi feita por validação cruzada; o teste só descreve o resultado.

## Como rodar

```bash
python -m venv .venv
.venv\Scripts\Activate.ps1          # Linux/macOS: source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip install -e .
# coloque train_FD001.txt, test_FD001.txt e RUL_FD001.txt em data/
python -m pytest
python -m cmapss.train_final --data-dir data --variant "+norm+ciclo" --tuning-csv results/tuning_final_norm_ciclo.csv
python app.py                       # demo local (precisa de: python -m pip install gradio)
```

## Estrutura
```
src/cmapss/     código (dados, variáveis, treino, experimentos, inferência, demo)
notebooks/      01 exploração · 02 análise de erros · 03 modelo final × base
tests/          testes automatizados
results/        tabelas de validação cruzada, ablação e ajuste
models/         modelo final salvo e metadados
examples/       motores de exemplo para a demo
app.py          demo em Gradio
```

## Próximos passos
- [ ] Testar outros subconjuntos (FD002–FD004, com várias condições de operação)
- [ ] Comparar com uma rede neural (LSTM/MLP)
- [ ] Atacar a regressão à média (por exemplo, perda assimétrica penalizando a superestimação)