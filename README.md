# Predição de vida útil de motores (NASA C-MAPSS)

> Preencha este README conforme avança. Substitua os trechos entre `[colchetes]`.

## Problema
Prever quantos ciclos de operação ainda restam (RUL, *Remaining Useful Life*) para cada motor turbofan a partir de leituras de sensores, para apoiar **manutenção preditiva**. Inspirado no meu projeto de monitoramento e manutenção preditiva do Grand Prix SENAI (2º lugar na América Latina).

## Dados
NASA C-MAPSS, subconjunto FD001: [100] motores de treino, [100] de teste, 21 sensores e 3 condições operacionais.

Dos 21 sensores, 6 são constantes (s1, s5, s10, s16, s18, s19) e foram descartados.
Os sensores s2, s3, s4, s8, s11, s13, s15 e s17 sobem perto da falha, e s7, s12, s20 e s21 descem.

![Sensores vs falha](reports/figures/02_sensores_vs_falha.png)

Com esses dados podemos ver como fica a distribuição da vida útil dos motores.

![Distribuição da vida útil](reports/figures/01_distribuicao_vida.png)

## Método
1. Exploração e remoção de sensores constantes (Feito)
2. Baseline: Regressão Linear e Random Forest   (Feito)
3. [Engenharia de variáveis: médias móveis, etc.] (em andamento)
4. [XGBoost e rede em PyTorch]
5. Validação separando por motor (sem vazamento entre treino e validação)

## Resultados (RMSE no teste, quanto menor, melhor)
## Resultados (RMSE no teste, quanto menor, melhor)

| Modelo | RMSE (teste, FD001) |
|---|---|
| Regressão Linear (baseline) | 20,82 |
| Random Forest (baseline) | 17,16 |
| XGBoost + médias móveis | em andamento |

O Random Forest reduziu o erro em cerca de 18% em relação à Regressão Linear (de 20,82 para 17,16 ciclos),
sem nenhum ajuste de hiperparâmetros. Isso sugere que a relação entre os sensores e o desgaste não é linear.
Ambos os modelos usam só a leitura do último ciclo de cada motor no teste, e os sensores constantes
(s1, s5, s10, s16, s18, s19) foram removidos. O próximo passo é incluir médias móveis dos sensores
para capturar a tendência de desgaste, e validar separando os dados por motor.

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
- [ ] [Adição de medias móveis]
- [ ] [modelar com XGBoost]
- [ ] [Validação por motor]
