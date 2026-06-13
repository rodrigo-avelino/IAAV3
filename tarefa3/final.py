"""
Analise final do GA Caixeiro Viajante: configuracao otima com mais geracoes.

Hiperparametros (validados nas etapas anteriores):
    N       = 200    (sintonia: melhor desempenho)
    K       = 5      (sintonia: pressao seletiva adequada)
    p_mut   = 0.01   (fixo pelo enunciado)
    N_elite = 0      (analise de elitismo: pressao seletiva ja suficiente,
                      elitismo causava convergencia prematura)
    max_gen = 5000   (acima do max_gen=1000 anterior; o teste exploratorio
                      mostrou que a curva continua descendo significativamente
                      ate ~5000 geracoes; depois disso o ganho marginal e
                      pequeno)

Justificativa do max_gen aumentado:
    O experimento anterior com max_gen=1000 mostrou que a curva do melhor
    individuo ainda decrescia ao final do orcamento de geracoes. Um teste
    exploratorio com 10000 geracoes revelou os seguintes ganhos relativos
    em relacao ao custo obtido em 1000 geracoes (seed unica):
        1000 -> 2000  geracoes:   -9%   (custo de 3298 para 2994)
        1000 -> 5000  geracoes:  -16%   (custo de 3298 para 2770)
        1000 -> 10000 geracoes:  -19%   (custo de 3298 para 2679)
    O ponto de retorno decrescente fica em torno de 5000 geracoes, que e
    o valor adotado nesta analise final.

Saida:
    resultados/final_5000ger.json
    resultados/final_5000ger.csv

Uso:
    python3 final.py            # roda as 30 rodadas (~23 min)
"""
import csv
import json
import os
import sys
import time
import numpy as np

from pontos import preparar_problema
from ga import ga_caixeiro


N_POP = 200
K_TORNEIO = 5
P_MUT = 0.01
N_ELITE = 0
MAX_GEN = 5000
N_RODADAS = 30
CSV_PATH = "CaixeiroGruposGA.csv"


def rodar(seed_base=20251203):
    cidades, grupos, D, n_cidades = preparar_problema(CSV_PATH)
    print(f"Problema: {n_cidades} cidades + origem")
    print(f"Config: N={N_POP}, K={K_TORNEIO}, p_mut={P_MUT}, "
          f"N_elite={N_ELITE}, max_gen={MAX_GEN}")
    print(f"Rodadas: {N_RODADAS}\n")

    rng_global = np.random.default_rng(seed_base)
    custos_finais = []
    hist_melhor_por_rodada = []
    tempos = []
    melhor_rota = None
    melhor_custo = np.inf

    t_total = time.time()
    for i in range(N_RODADAS):
        seed_rodada = int(rng_global.integers(0, 2**31 - 1))
        rng = np.random.default_rng(seed_rodada)
        t_r = time.time()
        r = ga_caixeiro(D, n_cidades, N=N_POP, max_gen=MAX_GEN,
                        p_mut=P_MUT, tamanho_torneio=K_TORNEIO,
                        N_elite=N_ELITE, rng=rng)
        dt = time.time() - t_r
        custos_finais.append(r["melhor_custo"])
        hist_melhor_por_rodada.append(r["hist_melhor"])
        tempos.append(dt)
        if r["melhor_custo"] < melhor_custo:
            melhor_custo = r["melhor_custo"]
            melhor_rota = r["melhor_x"].tolist()
        print(f"  rodada {i+1:2d}/{N_RODADAS}  custo={r['melhor_custo']:7.1f}  "
              f"tempo={dt:5.1f}s  acumulado={(time.time()-t_total)/60:5.1f}min")

    arr = np.array(custos_finais)
    resumo = {
        "N": N_POP, "K": K_TORNEIO, "p_mut": P_MUT,
        "N_elite": N_ELITE, "max_gen": MAX_GEN,
        "rodadas": N_RODADAS,
        "custos_finais": custos_finais,
        "media": float(arr.mean()), "desvio": float(arr.std()),
        "minimo": float(arr.min()), "maximo": float(arr.max()),
        "mediana": float(np.median(arr)),
        "tempo_total_segundos": time.time() - t_total,
        "tempo_por_rodada": tempos,
        "melhor_rota": melhor_rota,
        "hist_melhor_por_rodada": [list(h) for h in hist_melhor_por_rodada],
    }

    os.makedirs("resultados", exist_ok=True)
    with open("resultados/final_5000ger.json", "w") as fh:
        json.dump(resumo, fh, indent=2)
    with open("resultados/final_5000ger.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["rodada", "custo_final", "tempo_segundos"])
        for i, (c, t) in enumerate(zip(custos_finais, tempos)):
            w.writerow([i+1, f"{c:.2f}", f"{t:.1f}"])

    print(f"\n=== Resumo ===")
    print(f"  Media:   {arr.mean():.1f}")
    print(f"  Desvio:  {arr.std():.1f}")
    print(f"  Minimo:  {arr.min():.1f}")
    print(f"  Maximo:  {arr.max():.1f}")
    print(f"  Mediana: {np.median(arr):.1f}")
    print(f"  Tempo total: {(time.time()-t_total)/60:.1f} min")


if __name__ == "__main__":
    rodar()
