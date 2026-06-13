# Tarefa 4 - Noite final de testes

3 arquivos novos pra colocar na pasta tarefa4.

| Arquivo | O que faz |
|---|---|
| `sa_rastrigin_v2.py` | SA com sigma sintonizado (grid 3x3 de sigma x T0) |
| `ga_sigma_decrescente_v2.py` | Refinamento fino do sigma decrescente (grid 3x3 ao redor do vencedor) |
| `noite_extras.py` | Master script que roda os dois sequencialmente |

## Como rodar

```bash
cd tarefa4
python3 noite_extras.py
```

**Tempo estimado total: ~17 min** (SA v2: ~6min + sigma_dec v2: ~11min).

## O que cada um testa

### 1/2 - SA v2 (corrige sigma=0.5 inadequado)
Grid: sigma in {0.05, 0.10, 0.20} x T0 in {1, 10, 100}.
Hipotese: com sigma pequeno (como o LRS), SA deve ficar competitivo
e talvez supere o LRS (media 459) pela capacidade de escape via Metropolis.

### 2/2 - sigma decrescente v2 (refina o vencedor atual)
Grid: sigma_inicial in {0.7, 1.0, 1.5} x alpha_sigma in {0.9993, 0.9995, 0.9997}.
Hipotese: o (1.0, 0.9995) que deu 73 e o vencedor real, mas pode haver
config marginalmente melhor por perto.

## Quando voltar

Manda o zip dos resultados (so pasta `resultados/`) e eu monto:
- Heatmap SA (sigma x T0)
- Heatmap sigma_decrescente refinado
- Boxplot consolidado: LRS x GA fixo x SA x GA sigma_dec x GA proporcional
- Curvas de convergencia comparativas
- Analise final pra fechar a Tarefa 4
