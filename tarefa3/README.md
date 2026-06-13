# Tarefa 2.2 — GA Caixeiro Viajante 3D

## Estrutura
```
tarefa3_caixeiro/
├── CaixeiroGruposGA.csv   # arquivo de pontos
├── pontos.py              # carregador do CSV + matriz de distâncias
├── ga.py                  # algoritmo genético (OX vetorizado)
├── sintonia.py            # ETAPA 1: sintonia N × K        [JÁ FEITO]
├── elitismo.py            # ETAPA 2: análise de elitismo   [JÁ FEITO]
├── final.py               # ETAPA 3: rodada final 5000 ger [NOVO]
├── graficos.py            # gera todas as figuras
└── resultados/            # criado automaticamente
```

## ETAPA 3 — Análise final (PRÓXIMA)

Configuração validada nas etapas anteriores:
- N=200, K=5, p_mut=0.01, N_elite=0
- **max_gen=5000** (subido de 1000 — teste exploratório mostrou que a curva continua descendo)

Tempo estimado: **~23 min** (30 rodadas × ~46s cada).

```bash
cd tarefa3_caixeiro
python3 final.py
```

Vai imprimir cada rodada conforme termina e um resumo no final.

### O que me mandar depois

Zipa a pasta `resultados/` (mantém os arquivos da sintonia e elitismo) e manda. Vou pegar:
- `resultados/final_5000ger.json` (curvas + custos)

E rodo o `graficos.py` aqui pra gerar as figuras novas:
- `final_convergencia.png` — curva nova × curva antiga (1000 ger) sobrepostas
- `final_boxplot_comparativo.png` — boxplot 1000 ger vs 5000 ger
- `final_melhor_rota_3d.png` — a melhor rota encontrada
