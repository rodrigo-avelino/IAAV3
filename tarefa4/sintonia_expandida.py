"""
Sintonia EXPANDIDA do GA real-coded no Rastrigin n=50 (Tarefa 4).

Motivacao:
    A sintonia fina anterior teve N=10, sigma_mut=0.30 como melhor, mas
    esses valores estavam no extremo do grid. Apenas aumentar levemente
    transferiria o problema (provavelmente o novo extremo continuaria
    sendo o melhor). Esta etapa busca o "joelho da curva" indo BEM alem
    do otimo anterior, ate observar o desempenho parar de melhorar.

Grid expandido (cobertura ampla):
    N        in {2, 5, 10, 20}        -> ate N=2 (minimo pratico)
    sigma_mut in {0.50, 1.00, 2.00}    -> ate ~7x o antigo otimo

Demais hiperparametros mantidos:
    eta_SBX = 1, p_cross = 0.9, p_mut_gene = 1/n = 0.02, K = 3,
    sem elitismo, orcamento 100_000 avaliacoes.

Tempo estimado (validado em testes locais):
    Todas as configs rodam em ~5s/exec (avaliacao populacional vetorizada).
    12 configs x 30 rodadas x ~5s = ~30 min total.

Uso:
    python3 sintonia_expandida.py            # roda todas as 12 configs
"""
import csv
import json
import os
import time
import numpy as np

from funcao import rastrigin, INF, SUP, N_VARIAVEIS, convergiu
from ga_real import ga_real


N_VALORES = [2, 5, 10, 20]
SIGMA_MUT_VALORES = [0.50, 1.00, 2.00]
N_RODADAS = 30
ORCAMENTO = 100_000


def rodar(seed_base=20251208):
    print(f"Grid expandido: N x sigma_mut = {len(N_VALORES)} x "
          f"{len(SIGMA_MUT_VALORES)} = "
          f"{len(N_VALORES) * len(SIGMA_MUT_VALORES)} configs")
    print(f"Rodadas: {N_RODADAS} por config, orcamento {ORCAMENTO} avaliacoes\n")
    print(f"{'N':>4s}  {'sigma':>5s}  {'max_gen':>7s}  {'media':>7s}  {'desvio':>6s}  "
          f"{'min':>6s}  {'max':>6s}  {'tempo':>6s}")

    rng_global = np.random.default_rng(seed_base)
    resultados = []
    t_total = time.time()

    for N in N_VALORES:
        for sm in SIGMA_MUT_VALORES:
            max_gen = ORCAMENTO // N
            custos = []
            hists = []
            n_conv = 0
            t_cfg = time.time()
            for _ in range(N_RODADAS):
                seed = int(rng_global.integers(0, 2**31 - 1))
                rng = np.random.default_rng(seed)
                r = ga_real(rastrigin, N_VARIAVEIS, INF, SUP,
                            N=N, max_gen=max_gen, sigma_mut=sm, rng=rng)
                custos.append(r["melhor_custo"])
                hists.append(r["hist_melhor"])
                if convergiu(r["melhor_custo"]):
                    n_conv += 1
            arr = np.array(custos)
            reg = {
                "N": N, "sigma_mut": sm, "max_gen": max_gen,
                "custos_finais": custos,
                "media": float(arr.mean()), "desvio": float(arr.std()),
                "minimo": float(arr.min()), "maximo": float(arr.max()),
                "mediana": float(np.median(arr)),
                "taxa_convergencia": n_conv / N_RODADAS,
                "tempo_total_segundos": time.time() - t_cfg,
                "hist_melhor_por_rodada": [list(h) for h in hists],
            }
            resultados.append(reg)
            print(f"{N:4d}  {sm:5.2f}  {max_gen:7d}  "
                  f"{arr.mean():7.2f}  {arr.std():6.2f}  "
                  f"{arr.min():6.2f}  {arr.max():6.2f}  "
                  f"{time.time()-t_cfg:5.1f}s")

    melhor = min(resultados, key=lambda r: (r["media"], r["desvio"]))
    print(f"\nMELHOR CONFIG: N={melhor['N']}, sigma_mut={melhor['sigma_mut']}  "
          f"(media={melhor['media']:.2f}, desvio={melhor['desvio']:.2f}, "
          f"min={melhor['minimo']:.2f})")
    print(f"Tempo total: {(time.time()-t_total)/60:.1f} min")

    os.makedirs("resultados", exist_ok=True)
    with open("resultados/sintonia_expandida.json", "w") as fh:
        json.dump({"configs": resultados, "melhor": melhor}, fh, indent=2)
    with open("resultados/sintonia_expandida.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["N", "sigma_mut", "max_gen", "media", "desvio",
                    "minimo", "maximo", "mediana", "taxa_convergencia",
                    "tempo_segundos"])
        for r in resultados:
            w.writerow([r["N"], r["sigma_mut"], r["max_gen"],
                        f"{r['media']:.3f}", f"{r['desvio']:.3f}",
                        f"{r['minimo']:.3f}", f"{r['maximo']:.3f}",
                        f"{r['mediana']:.3f}", f"{r['taxa_convergencia']:.3f}",
                        f"{r['tempo_total_segundos']:.1f}"])


if __name__ == "__main__":
    rodar()
