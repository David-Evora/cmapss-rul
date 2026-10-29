# Predição de vida útil de motores (NASA C-MAPSS)

> Preencha este README conforme avança. Substitua os trechos entre `[colchetes]`.

## Problema
Prever quantos ciclos de operação ainda restam (RUL, *Remaining Useful Life*) para cada motor turbofan a partir de leituras de sensores, para apoiar **manutenção preditiva**. Inspirado no meu projeto de monitoramento e manutenção preditiva do Grand Prix SENAI (2º lugar na América Latina).

## Dados
NASA C-MAPSS, subconjunto FD001: [N] motores de treino, [N] de teste, 21 sensores e 3 condições operacionais.

## Método
1. Exploração e remoção de sensores constantes
2. Baseline: Regressão Linear e Random Forest
3. [Engenharia de variáveis: médias móveis, etc.]
4. [XGBoost e rede em PyTorch]
5. Validação separando por motor (sem vazamento entre treino e validação)

## Resultados (RMSE no teste, quanto menor, melhor)

| Modelo | RMSE (teste, FD001) |
|---|---|
| Regressão Linear (baseline) | 20,82 |
| Random Forest (baseline) | 17,16 |
| [Modelo final] | [em andamento] |

[Um parágrafo sobre o que funcionou, o que não funcionou e onde o modelo erra mais.]

## Como rodar

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
# coloque train_FD001.txt, test_FD001.txt e RUL_FD001.txt em data/
python -m pytest
PYTHONPATH=src python -m cmapss.train_baseline --data-dir data
```

## Estrutura
```
src/cmapss/     código (carregamento de dados, treino)
notebooks/      exploração e experimentos
tests/          testes
results/        métricas geradas
```

## Próximos passos
- [ ] [ideia de melhoria]
