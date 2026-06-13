"""
Sintonia da Tempera Simulada (SA) no Rastrigin n=50.

Compara contra o LRS e o GA como terceira via metodologica.
Memoria O(n) identica a LRS; mesmo orcamento de avaliacoes.

Hiperparametros sintonizados:
    T0 in {10, 100, 1000}   (temperatura inicial)

Fixos:
    alpha = 0.9999  (resfriamento geometrico; em 100k it, T_final ~ 4.5e-5 * T0)
    sigma = 0.5     (mesma faixa do LRS sintonizado)
    max_it = 100000 (orcamento de avaliacoes)

Tempo estimado: ~2.5 min total.
"""
import csv
import json
import os
import time
import numpy as np

from funcao import rastrigin, INF, SUP, N_VARIAVEIS, convergiu
from busca_local import sa


T0_VALORES = [10.0, 100.0, 1000.0]
ALPHA = 0.9999
SIGMA = 0.5
MAX_IT = 100_000
N_RODADAS = 30


def rodar(seed_base=20251212):
    print(f"Sintonia SA Rastrigin: T0 in {T0_VALORES}, alpha={ALPHA}, sigma={SIGMA}")
    print(f"{N_RODADAS} rodadas, max_it={MAX_IT}\n")
    print(f"{'T0':>7s}  {'media':>7s}  {'desvio':>6s}  {'min':>6s}  "
          f"{'max':>6s}  {'aceit':>6s}  {'tempo':>5s}")

    rng_global = np.random.default_rng(seed_base)
    resultados = []
    t_total = time.time()

    for T0 in T0_VALORES:
        custos = []
        hists = []
        taxas = []
        n_conv = 0
        t_cfg = time.time()
        for _ in range(N_RODADAS):
            seed = int(rng_global.integers(0, 2**31 - 1))
            rng = np.random.default_rng(seed)
            r = sa(rastrigin, N_VARIAVEIS, INF, SUP,
                   sigma=SIGMA, T0=T0, alpha=ALPHA, max_it=MAX_IT, rng=rng)
            custos.append(r["melhor_custo"])
            hists.append(r["hist_melhor"])
            taxas.append(r["taxa_aceitacao"])
            if convergiu(r["melhor_custo"]):
                n_conv += 1
        arr = np.array(custos)
        reg = {
            "T0": T0, "alpha": ALPHA, "sigma": SIGMA,
            "custos_finais": custos,
            "media": float(arr.mean()), "desvio": float(arr.std()),
            "minimo": float(arr.min()), "maximo": float(arr.max()),
            "mediana": float(np.median(arr)),
            "taxa_aceitacao_media": float(np.mean(taxas)),
            "taxa_convergencia": n_conv / N_RODADAS,
            "tempo_total_segundos": time.time() - t_cfg,
            "hist_melhor_por_rodada": [list(h) for h in hists],
        }
        resultados.append(reg)
        print(f"{T0:7.1f}  {arr.mean():7.2f}  {arr.std():6.2f}  "
              f"{arr.min():6.2f}  {arr.max():6.2f}  "
              f"{np.mean(taxas):6.2%}  {time.time()-t_cfg:5.1f}s")

    melhor = min(resultados, key=lambda r: (r["media"], r["desvio"]))
    print(f"\nMELHOR: T0={melhor['T0']}  "
          f"media={melhor['media']:.2f}  desvio={melhor['desvio']:.2f}")
    print(f"Tempo total: {(time.time()-t_total)/60:.1f} min")

    os.makedirs("resultados", exist_ok=True)
    with open("resultados/sa_sintonia.json", "w") as fh:
        json.dump({"configs": resultados, "melhor": melhor}, fh, indent=2)
    with open("resultados/sa_sintonia.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["T0", "media", "desvio", "minimo", "maximo", "mediana",
                    "taxa_aceitacao_media", "taxa_convergencia", "tempo_segundos"])
        for r in resultados:
            w.writerow([r["T0"],
                        f"{r['media']:.3f}", f"{r['desvio']:.3f}",
                        f"{r['minimo']:.3f}", f"{r['maximo']:.3f}",
                        f"{r['mediana']:.3f}",
                        f"{r['taxa_aceitacao_media']:.3f}",
                        f"{r['taxa_convergencia']:.3f}",
                        f"{r['tempo_total_segundos']:.1f}"])


if __name__ == "__main__":
    rodar()
