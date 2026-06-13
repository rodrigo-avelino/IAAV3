"""
GA sigma decrescente v2: refinamento fino em torno do otimo.

A sintonia inicial identificou (sigma_inicial=1.0, alpha=0.9995) como otima
com media 73. Esta etapa refina em torno desse ponto para verificar se nao
ha config marginalmente melhor por perto.

Grid refinado:
    sigma_inicial in {0.7, 1.0, 1.5}
    alpha_sigma  in {0.9993, 0.9995, 0.9997}

Fixos:
    N = 10, max_gen = 10000, demais hp iguais ao GA padrao.

Tempo estimado: ~11 min (9 configs x 30 rodadas x ~2.4s/exec).
"""
import csv
import json
import os
import time
import numpy as np

from funcao import rastrigin, INF, SUP, N_VARIAVEIS, convergiu
from ga_real import ga_real


SIGMA_INICIAL_VALORES = [0.7, 1.0, 1.5]
ALPHA_SIGMA_VALORES = [0.9993, 0.9995, 0.9997]
N_POP = 10
MAX_GEN = 10_000
N_RODADAS = 30


def rodar(seed_base=20251216):
    print(f"GA sigma decrescente v2 (refinamento): "
          f"sigma_inicial x alpha_sigma = "
          f"{len(SIGMA_INICIAL_VALORES)} x {len(ALPHA_SIGMA_VALORES)} = "
          f"{len(SIGMA_INICIAL_VALORES) * len(ALPHA_SIGMA_VALORES)} configs")
    print(f"N={N_POP}, max_gen={MAX_GEN}, {N_RODADAS} rodadas\n")
    print(f"{'sig0':>5s}  {'alpha':>7s}  {'sig_fim':>7s}  {'media':>7s}  "
          f"{'desvio':>6s}  {'min':>6s}  {'tempo':>5s}")

    rng_global = np.random.default_rng(seed_base)
    resultados = []
    t_total = time.time()

    for sigma_inicial in SIGMA_INICIAL_VALORES:
        for alpha_sigma in ALPHA_SIGMA_VALORES:
            custos = []
            hists = []
            hists_sigma = []
            n_conv = 0
            t_cfg = time.time()
            for _ in range(N_RODADAS):
                seed = int(rng_global.integers(0, 2**31 - 1))
                rng = np.random.default_rng(seed)
                r = ga_real(rastrigin, N_VARIAVEIS, INF, SUP,
                            N=N_POP, max_gen=MAX_GEN,
                            sigma_mut=sigma_inicial, alpha_sigma=alpha_sigma,
                            rng=rng)
                custos.append(r["melhor_custo"])
                hists.append(r["hist_melhor"])
                hists_sigma.append(r["hist_sigma"])
                if convergiu(r["melhor_custo"]):
                    n_conv += 1
            arr = np.array(custos)
            sigma_final = sigma_inicial * (alpha_sigma ** MAX_GEN)
            reg = {
                "sigma_inicial": sigma_inicial, "alpha_sigma": alpha_sigma,
                "sigma_final_esperado": float(sigma_final),
                "N": N_POP, "max_gen": MAX_GEN,
                "custos_finais": custos,
                "media": float(arr.mean()), "desvio": float(arr.std()),
                "minimo": float(arr.min()), "maximo": float(arr.max()),
                "mediana": float(np.median(arr)),
                "taxa_convergencia": n_conv / N_RODADAS,
                "tempo_total_segundos": time.time() - t_cfg,
                "hist_melhor_por_rodada": [list(h) for h in hists],
                "hist_sigma_por_rodada": [list(h) for h in hists_sigma],
            }
            resultados.append(reg)
            print(f"{sigma_inicial:5.2f}  {alpha_sigma:7.4f}  "
                  f"{sigma_final:7.4f}  {arr.mean():7.2f}  "
                  f"{arr.std():6.2f}  {arr.min():6.2f}  "
                  f"{time.time()-t_cfg:5.1f}s")

    melhor = min(resultados, key=lambda r: (r["media"], r["desvio"]))
    print(f"\nMELHOR: sigma_inicial={melhor['sigma_inicial']}, "
          f"alpha_sigma={melhor['alpha_sigma']}  "
          f"(media={melhor['media']:.2f}, desvio={melhor['desvio']:.2f})")
    print(f"Tempo total: {(time.time()-t_total)/60:.1f} min")

    os.makedirs("resultados", exist_ok=True)
    with open("resultados/ga_sigma_decrescente_v2.json", "w") as fh:
        json.dump({"configs": resultados, "melhor": melhor}, fh, indent=2)
    with open("resultados/ga_sigma_decrescente_v2.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["sigma_inicial", "alpha_sigma", "sigma_final_esperado",
                    "media", "desvio", "minimo", "maximo", "mediana",
                    "taxa_convergencia", "tempo_segundos"])
        for r in resultados:
            w.writerow([r["sigma_inicial"], r["alpha_sigma"],
                        f"{r['sigma_final_esperado']:.6f}",
                        f"{r['media']:.3f}", f"{r['desvio']:.3f}",
                        f"{r['minimo']:.3f}", f"{r['maximo']:.3f}",
                        f"{r['mediana']:.3f}",
                        f"{r['taxa_convergencia']:.3f}",
                        f"{r['tempo_total_segundos']:.1f}"])


if __name__ == "__main__":
    rodar()
