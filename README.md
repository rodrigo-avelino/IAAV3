# Expansão T3 e T4 — buscar o "joelho da curva"

## Mudança de estratégia

Os anteriores tinham o melhor parâmetro **no extremo do grid**, então
expandir um pouco só transferiria o problema. Esta versão vai bem além
para descobrir **onde o desempenho atinge o pico e começa a piorar**.

## Arquivos

| Arquivo | Status |
|---|---|
| `tarefa3/sintonia_expandida.py` | NOVO/REVISADO (grid mais amplo) |
| `tarefa3/graficos.py` | MODIFICADO (heatmap combinado original + expandido) |
| `tarefa4/sintonia_expandida.py` | NOVO/REVISADO (grid mais amplo) |
| `tarefa4/graficos.py` | MODIFICADO (heatmap combinado fina + expandida) |
| `tarefa4/ga_real.py` | CORRIGIDO (bug quando K > N na seleção por torneio) |

## T3 — caixeiro viajante

Grid: N ∈ {500, 1000, 2000} × K ∈ {5, 15}. 6 configs total.

**Tempo: ~2h sequencial** (N=500 ~18min, N=1000 ~35min, N=2000 ~70min).

```bash
cd tarefa3
python3 sintonia_expandida.py
```

Ou em partes:
```bash
python3 sintonia_expandida.py 500    # ~18 min
python3 sintonia_expandida.py 1000   # ~35 min
python3 sintonia_expandida.py 2000   # ~70 min
python3 sintonia_expandida.py agregar
```

## T4 — Rastrigin

Grid: N ∈ {2, 5, 10, 20} × σ_mut ∈ {0.5, 1.0, 2.0}. 12 configs total.

**Tempo: ~30 min** (todas as configs rodam em ~5s/exec graças à vetorização).

```bash
cd tarefa4
python3 sintonia_expandida.py
```

## O que me mandar depois

Zipa as pastas `resultados/` (T3 e T4) e manda. Eu gero:
- Heatmap combinado (grid original + expandido) das duas tarefas
- Boxplot atualizado
- Curvas de convergência
- Análise: o joelho está dentro do grid? Se não, expandimos mais.
