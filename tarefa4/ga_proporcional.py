"""
Sintonia do GA real-coded com mutacao inversamente proporcional a aptidao.

Cada individuo recebe um sigma local proporcional ao seu rank na populacao:
    sigma_local = sigma_min + (sigma_max - sigma_min) * rank/(N-1)
onde rank=0 (melhor) recebe sigma_min e rank=N-1 (pior) recebe sigma_max.

Hiperparametros sintonizados:
    sigma_max in {0.5, 1.0, 2.0}   (mutacao agressiva dos piores)

Fixos:
    sigma_min = 0.05   (mutacao quase nula dos melhores)
    N = 10, K = 3, p_mut_gene = 1/n = 0.02, p_cross = 0.9, eta_SBX = 1.0
    max_gen = 10000 (orcamento 100k avaliacoes)
"""
import csv
import json
import os
import time
import numpy as np

from funcao import rastrigin, INF, SUP, N_VARIAVEIS, convergiu
from ga_real import ga_real


SIGMA_MIN = 0.05
SIGMA_MAX_VALORES = [0.5, 1.0, 2.0]
N_POP = 10
MAX_GEN = 10_000
N_RODADAS = 30


def rodar(seed_base=20251214):
    print(f"Sintonia GA mutacao proporcional a aptidao: "
          f"sigma_max in {SIGMA_MAX_VALORES} (sigma_min={SIGMA_MIN} fixo)")
    print(f"N={N_POP}, max_gen={MAX_GEN}, {N_RODADAS} rodadas\n")
    print(f"{'s_min':>6s}  {'s_max':>6s}  {'media':>7s}  {'desvio':>6s}  "
          f"{'min':>6s}  {'tempo':>5s}")

    rng_global = np.random.default_rng(seed_base)
    resultados = []
    t_total = time.time()

    for sigma_max in SIGMA_MAX_VALORES:
        custos = []
        hists = []
        n_conv = 0
        t_cfg = time.time()
        for _ in range(N_RODADAS):
            seed = int(rng_global.integers(0, 2**31 - 1))
            rng = np.random.default_rng(seed)
            r = ga_real(rastrigin, N_VARIAVEIS, INF, SUP,
                        N=N_POP, max_gen=MAX_GEN,
                        sigma_proporcional=(SIGMA_MIN, sigma_max),
                        rng=rng)
            custos.append(r["melhor_custo"])
            hists.append(r["hist_melhor"])
            if convergiu(r["melhor_custo"]):
                n_conv += 1
        arr = np.array(custos)
        reg = {
            "sigma_min": SIGMA_MIN, "sigma_max": sigma_max,
            "N": N_POP, "max_gen": MAX_GEN,
            "custos_finais": custos,
            "media": float(arr.mean()), "desvio": float(arr.std()),
            "minimo": float(arr.min()), "maximo": float(arr.max()),
            "mediana": float(np.median(arr)),
            "taxa_convergencia": n_conv / N_RODADAS,
            "tempo_total_segundos": time.time() - t_cfg,
            "hist_melhor_por_rodada": [list(h) for h in hists],
        }
        resultados.append(reg)
        print(f"{SIGMA_MIN:6.3f}  {sigma_max:6.2f}  {arr.mean():7.2f}  "
              f"{arr.std():6.2f}  {arr.min():6.2f}  {time.time()-t_cfg:5.1f}s")

    melhor = min(resultados, key=lambda r: (r["media"], r["desvio"]))
    print(f"\nMELHOR: sigma_max={melhor['sigma_max']}  "
          f"(media={melhor['media']:.2f}, desvio={melhor['desvio']:.2f})")
    print(f"Tempo total: {(time.time()-t_total)/60:.1f} min")

    os.makedirs("resultados", exist_ok=True)
    with open("resultados/ga_proporcional.json", "w") as fh:
        json.dump({"configs": resultados, "melhor": melhor}, fh, indent=2)
    with open("resultados/ga_proporcional.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["sigma_min", "sigma_max",
                    "media", "desvio", "minimo", "maximo", "mediana",
                    "taxa_convergencia", "tempo_segundos"])
        for r in resultados:
            w.writerow([r["sigma_min"], r["sigma_max"],
                        f"{r['media']:.3f}", f"{r['desvio']:.3f}",
                        f"{r['minimo']:.3f}", f"{r['maximo']:.3f}",
                        f"{r['mediana']:.3f}",
                        f"{r['taxa_convergencia']:.3f}",
                        f"{r['tempo_total_segundos']:.1f}"])


if __name__ == "__main__":
    rodar()
