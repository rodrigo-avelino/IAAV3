"""
SA v2: sintonia conjunta de sigma e T0 no Rastrigin n=50.

Motivacao:
    Na sintonia inicial, sigma=0.5 ficou fixado (heranca do LRS), mas o
    LRS sintonizado venceu com sigma=0.05. Testes anteriores mostraram
    que SA com sigma=0.5 fica com taxa_aceitacao ~0% para T0 baixo,
    indicando que o algoritmo degenera em HC. Esta etapa testa SA com
    sigma adequado.

Grid:
    sigma in {0.05, 0.10, 0.20}
    T0    in {1.0, 10.0, 100.0}

Fixos:
    alpha = 0.9999, max_it = 100000

Tempo estimado: ~6 min (9 configs x 30 rodadas x ~1.3s/exec).
"""
import csv
import json
import os
import time
import numpy as np

from funcao import rastrigin, INF, SUP, N_VARIAVEIS, convergiu
from busca_local import sa


SIGMA_VALORES = [0.05, 0.10, 0.20]
T0_VALORES = [1.0, 10.0, 100.0]
ALPHA = 0.9999
MAX_IT = 100_000
N_RODADAS = 30


def rodar(seed_base=20251215):
    print(f"SA v2: sigma x T0 = {len(SIGMA_VALORES)} x {len(T0_VALORES)} = "
          f"{len(SIGMA_VALORES) * len(T0_VALORES)} configs")
    print(f"{N_RODADAS} rodadas, alpha={ALPHA}, max_it={MAX_IT}\n")
    print(f"{'sigma':>6s}  {'T0':>6s}  {'media':>7s}  {'desvio':>6s}  "
          f"{'min':>6s}  {'aceit':>6s}  {'tempo':>5s}")

    rng_global = np.random.default_rng(seed_base)
    resultados = []
    t_total = time.time()

    for sigma in SIGMA_VALORES:
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
                       sigma=sigma, T0=T0, alpha=ALPHA,
                       max_it=MAX_IT, rng=rng)
                custos.append(r["melhor_custo"])
                hists.append(r["hist_melhor"])
                taxas.append(r["taxa_aceitacao"])
                if convergiu(r["melhor_custo"]):
                    n_conv += 1
            arr = np.array(custos)
            reg = {
                "sigma": sigma, "T0": T0, "alpha": ALPHA,
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
            print(f"{sigma:6.2f}  {T0:6.1f}  {arr.mean():7.2f}  "
                  f"{arr.std():6.2f}  {arr.min():6.2f}  "
                  f"{np.mean(taxas):6.2%}  {time.time()-t_cfg:5.1f}s")

    melhor = min(resultados, key=lambda r: (r["media"], r["desvio"]))
    print(f"\nMELHOR: sigma={melhor['sigma']}, T0={melhor['T0']}  "
          f"(media={melhor['media']:.2f}, desvio={melhor['desvio']:.2f})")
    print(f"Tempo total: {(time.time()-t_total)/60:.1f} min")

    os.makedirs("resultados", exist_ok=True)
    with open("resultados/sa_sintonia_v2.json", "w") as fh:
        json.dump({"configs": resultados, "melhor": melhor}, fh, indent=2)
    with open("resultados/sa_sintonia_v2.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["sigma", "T0", "media", "desvio", "minimo", "maximo",
                    "mediana", "taxa_aceitacao_media", "taxa_convergencia",
                    "tempo_segundos"])
        for r in resultados:
            w.writerow([r["sigma"], r["T0"],
                        f"{r['media']:.3f}", f"{r['desvio']:.3f}",
                        f"{r['minimo']:.3f}", f"{r['maximo']:.3f}",
                        f"{r['mediana']:.3f}",
                        f"{r['taxa_aceitacao_media']:.3f}",
                        f"{r['taxa_convergencia']:.3f}",
                        f"{r['tempo_total_segundos']:.1f}"])


if __name__ == "__main__":
    rodar()
