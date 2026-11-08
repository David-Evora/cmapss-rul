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
| Rede neural | GRU e LSTM sobre janelas de 30 ciclos, 5 sementes, com e sem o número do ciclo como entrada; comparadas ao XGBoost final (notebook 04) |
| Entrega | Modelo salvo, demo em Gradio e publicação no Hugging Face (`app.py`, `src/cmapss/publish_hf.py`) |

Não se aplicam a este conjunto: imputação de valores ausentes, codificação de categorias e remoção de outliers (os dados são numéricos, completos e simulados).

## Resultados (último ciclo de cada um dos 100 motores de teste)

| Modelo | RMSE (real ≤125) |
|---|---|
| Regressão Linear (baseline) | 20,82 |
| Random Forest (baseline) | 17,16 |
| XGBoost ajustado, variáveis base | ≈16 |
| GRU, janela de 30 ciclos (média de 5 sementes) | 13,72 ± 0,10 |
| LSTM, janela de 30 ciclos (média de 5 sementes) | 13,83 ± 0,54 |
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
- As redes GRU e LSTM não superaram o XGBoost com variáveis construídas, nem recebendo o número do ciclo (detalhes abaixo).
- Dados simulados, uma única condição de operação (FD001) e alvo limitado a 125 ciclos: não é validação para motores reais.
- A escolha das variáveis e dos hiperparâmetros foi feita por validação cruzada; o teste só descreve o resultado.

## Rede neural (notebook 04)

Pergunta: uma rede recorrente que enxerga os últimos 30 ciclos de cada motor bate o XGBoost final?

**Configuração.** Entrada: janelas de 30 ciclos (passo 1) dos 14 sensores que variam, padronizados com estatísticas do treino. Alvo: RUL limitada a 125. Rede: 2 camadas recorrentes de 64 unidades (dropout 0,2) e uma cabeça densa; Adam com taxa de aprendizado 1e-3, até 80 épocas, parada antecipada com paciência de 12. Validação: 80% dos motores para treino e 20% para validação (por motor, para não vazar janelas parecidas entre treino e validação); o teste não participa da parada antecipada. Avaliação: última janela de cada um dos 100 motores de teste, 5 sementes (0 a 4). A variante `sensores+ciclo` acrescenta o número do ciclo a cada passo da janela, porque o XGBoost final usa essa informação e uma janela de 30 ciclos sozinha não conhece a idade absoluta do motor.

| Modelo | RMSE (≤125) | RMSE (sem teto) | Score assimétrico (médio) |
|---|---|---|---|
| GRU, sensores | 13,72 ± 0,10 | 15,06 | 374,3 |
| LSTM, sensores | 13,83 ± 0,54 | 15,02 | 392,9 |
| GRU, sensores + ciclo | 14,02 ± 1,08 | 15,45 | 402,0 |
| XGBoost final (referência, seed 0) | 10,98 | 12,12 | 171,6 |

Média e desvio-padrão entre as 5 sementes. A média de sementes do XGBoost final é 11,15 ± 0,10 (≤125).

![RMSE por semente e previsto vs real](reports/figures/04_rnn_fd001.png)

**Conclusão.** Nenhuma das variantes se aproxima do XGBoost final. Dar o número do ciclo à rede melhorou o erro de validação (12,15 contra 12,85 para a GRU), mas não o de teste, e deixou o resultado instável (uma das sementes chegou a 15,93). Com apenas 100 motores de treino, as variáveis construídas à mão (normalização pelo nível inicial do motor, inclinações, ciclo) foram mais eficientes do que a rede aprender a representação sozinha. O erro de validação da rede é calculado sobre todas as janelas dos motores de validação, então não é diretamente comparável com o erro de teste (última janela de cada motor).

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

O notebook 04 (GRU/LSTM) precisa de PyTorch (`python -m pip install torch`) e leva cerca de 30 minutos em CPU para as 5 sementes e as 3 variantes.

## Estrutura
```
src/cmapss/     código (dados, variáveis, treino, experimentos, inferência, demo)
notebooks/      01 exploração · 02 análise de erros · 03 modelo final × base · 04 GRU/LSTM
tests/          testes automatizados
results/        tabelas de validação cruzada, ablação, ajuste e rede neural
figures/        gráficos do notebook 04
models/         modelo final salvo e metadados
examples/       motores de exemplo para a demo
app.py          demo em Gradio
```

## Próximos passos
- [ ] Testar outros subconjuntos (FD002–FD004, com várias condições de operação)
- [x] Comparar com uma rede neural (GRU/LSTM): não supera o XGBoost final (notebook 04)
- [ ] Atacar a regressão à média (por exemplo, perda assimétrica penalizando a superestimação)
