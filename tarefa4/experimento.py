"""
Experimento principal da Tarefa 2.3: comparacao GA real-coded vs LRS
no Rastrigin n=50.

Comparacao justa por ORCAMENTO DE AVALIACOES DA FUNCAO:
    - Cada metodo tem orcamento de 100_000 avaliacoes por rodada.
    - LRS:  max_it = 100_000   (1 avaliacao por iteracao)
    - GA:   max_gen = floor(100_000 / N)
            (N avaliacoes por geracao + N iniciais)
    Memoria do GA (proporcional a N) e maior que a do LRS (constante).

Sintonia:
    LRS varia o sigma da perturbacao gaussiana.
        Candidatos: {0.05, 0.10, 0.20, 0.50, 1.00}
    GA varia o tamanho da populacao N, sintonizando indiretamente o numero
    de geracoes (devido ao orcamento fixo).
        Candidatos: {30, 50, 100, 200}

Criterio: menor custo medio em 30 rodadas; desempate por menor desvio.

Uso:
    python3 experimento.py            # roda tudo (~15 min)
    python3 experimento.py lrs        # so a sintonia do LRS
    python3 experimento.py ga         # so a sintonia do GA
"""
import csv
import json
import os
import sys
import time
import numpy as np

from funcao import rastrigin, INF, SUP, N_VARIAVEIS, OTIMO, TOLERANCIA, convergiu
from busca_local import lrs
from ga_real import ga_real


N_RODADAS = 30
ORCAMENTO_AVALIACOES = 100_000

SIGMAS_LRS = [0.05, 0.10, 0.20, 0.50, 1.00]
NS_GA = [30, 50, 100, 200]


def sintonia_lrs(seed_base=20251204):
    print(f"\n=== Sintonia LRS (orcamento {ORCAMENTO_AVALIACOES} aval/rodada) ===")
    rng_global = np.random.default_rng(seed_base)
    resultados = []
    for sigma in SIGMAS_LRS:
        custos = []
        hists = []
        n_conv = 0
        t_cfg = time.time()
        for _ in range(N_RODADAS):
            seed = int(rng_global.integers(0, 2**31 - 1))
            rng = np.random.default_rng(seed)
            r = lrs(rastrigin, N_VARIAVEIS, INF, SUP,
                    sigma=sigma, max_it=ORCAMENTO_AVALIACOES, rng=rng)
            custos.append(r["melhor_custo"])
            hists.append(r["hist_melhor"])
            if convergiu(r["melhor_custo"]):
                n_conv += 1
        arr = np.array(custos)
        reg = {
            "sigma": sigma,
            "custos_finais": custos,
            "media": float(arr.mean()), "desvio": float(arr.std()),
            "minimo": float(arr.min()), "maximo": float(arr.max()),
            "mediana": float(np.median(arr)),
            "taxa_convergencia": n_conv / N_RODADAS,
            "tempo_total_segundos": time.time() - t_cfg,
            "hist_melhor_por_rodada": [list(h) for h in hists],
        }
        resultados.append(reg)
        print(f"  sigma={sigma:4.2f}  media={arr.mean():7.2f}  "
              f"desvio={arr.std():6.2f}  min={arr.min():6.2f}  "
              f"taxa_conv={n_conv*100/N_RODADAS:5.1f}%  tempo={time.time()-t_cfg:5.1f}s")
    return resultados


def sintonia_ga(seed_base=20251205):
    print(f"\n=== Sintonia GA (orcamento {ORCAMENTO_AVALIACOES} aval/rodada) ===")
    rng_global = np.random.default_rng(seed_base)
    resultados = []
    for N in NS_GA:
        max_gen = ORCAMENTO_AVALIACOES // N
        custos = []
        hists = []
        n_conv = 0
        t_cfg = time.time()
        for _ in range(N_RODADAS):
            seed = int(rng_global.integers(0, 2**31 - 1))
            rng = np.random.default_rng(seed)
            r = ga_real(rastrigin, N_VARIAVEIS, INF, SUP,
                        N=N, max_gen=max_gen, rng=rng)
            custos.append(r["melhor_custo"])
            hists.append(r["hist_melhor"])
            if convergiu(r["melhor_custo"]):
                n_conv += 1
        arr = np.array(custos)
        reg = {
            "N": N, "max_gen": max_gen,
            "custos_finais": custos,
            "media": float(arr.mean()), "desvio": float(arr.std()),
            "minimo": float(arr.min()), "maximo": float(arr.max()),
            "mediana": float(np.median(arr)),
            "taxa_convergencia": n_conv / N_RODADAS,
            "tempo_total_segundos": time.time() - t_cfg,
            "hist_melhor_por_rodada": [list(h) for h in hists],
        }
        resultados.append(reg)
        print(f"  N={N:3d}  max_gen={max_gen:5d}  media={arr.mean():7.2f}  "
              f"desvio={arr.std():6.2f}  min={arr.min():6.2f}  "
              f"taxa_conv={n_conv*100/N_RODADAS:5.1f}%  tempo={time.time()-t_cfg:5.1f}s")
    return resultados


def salvar(resultados_lrs=None, resultados_ga=None):
    os.makedirs("resultados", exist_ok=True)

    if resultados_lrs is not None:
        with open("resultados/lrs_sintonia.json", "w") as fh:
            json.dump({"configs": resultados_lrs}, fh, indent=2)
        with open("resultados/lrs_sintonia.csv", "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["sigma", "media", "desvio", "minimo", "maximo",
                        "mediana", "taxa_convergencia", "tempo_segundos"])
            for r in resultados_lrs:
                w.writerow([r["sigma"],
                            f"{r['media']:.3f}", f"{r['desvio']:.3f}",
                            f"{r['minimo']:.3f}", f"{r['maximo']:.3f}",
                            f"{r['mediana']:.3f}", f"{r['taxa_convergencia']:.3f}",
                            f"{r['tempo_total_segundos']:.1f}"])

    if resultados_ga is not None:
        with open("resultados/ga_sintonia.json", "w") as fh:
            json.dump({"configs": resultados_ga}, fh, indent=2)
        with open("resultados/ga_sintonia.csv", "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["N", "max_gen", "media", "desvio", "minimo", "maximo",
                        "mediana", "taxa_convergencia", "tempo_segundos"])
            for r in resultados_ga:
                w.writerow([r["N"], r["max_gen"],
                            f"{r['media']:.3f}", f"{r['desvio']:.3f}",
                            f"{r['minimo']:.3f}", f"{r['maximo']:.3f}",
                            f"{r['mediana']:.3f}", f"{r['taxa_convergencia']:.3f}",
                            f"{r['tempo_total_segundos']:.1f}"])


def imprimir_comparacao(resultados_lrs, resultados_ga):
    melhor_lrs = min(resultados_lrs, key=lambda r: (r["media"], r["desvio"]))
    melhor_ga = min(resultados_ga, key=lambda r: (r["media"], r["desvio"]))
    print("\n=== COMPARACAO FINAL ===")
    print(f"  Melhor LRS: sigma={melhor_lrs['sigma']}  "
          f"media={melhor_lrs['media']:.2f}  desvio={melhor_lrs['desvio']:.2f}  "
          f"min={melhor_lrs['minimo']:.2f}")
    print(f"  Melhor GA:  N={melhor_ga['N']}, max_gen={melhor_ga['max_gen']}  "
          f"media={melhor_ga['media']:.2f}  desvio={melhor_ga['desvio']:.2f}  "
          f"min={melhor_ga['minimo']:.2f}")
    if melhor_lrs["media"] < melhor_ga["media"]:
        print(f"  --> LRS vence por {melhor_ga['media'] - melhor_lrs['media']:.2f}")
    else:
        print(f"  --> GA vence por {melhor_lrs['media'] - melhor_ga['media']:.2f}")


if __name__ == "__main__":
    t0 = time.time()
    if len(sys.argv) > 1 and sys.argv[1] == "lrs":
        res = sintonia_lrs()
        salvar(resultados_lrs=res)
    elif len(sys.argv) > 1 and sys.argv[1] == "ga":
        res = sintonia_ga()
        salvar(resultados_ga=res)
    else:
        res_lrs = sintonia_lrs()
        res_ga = sintonia_ga()
        salvar(resultados_lrs=res_lrs, resultados_ga=res_ga)
        imprimir_comparacao(res_lrs, res_ga)
    print(f"\nTempo total: {(time.time()-t0)/60:.1f} min")
