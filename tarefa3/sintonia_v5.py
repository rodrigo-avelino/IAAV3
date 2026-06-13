"""
Sintonia v5 do GA caixeiro viajante: confirmacao final do joelho.

Motivacao:
    O patamar N=8000 mostrou ganho marginal de -2.4% (vs -3.2% dos
    anteriores) e o minimo observado praticamente nao mudou em relacao
    a N=5000. O desvio-padrao caiu pela metade (86 -> 50), sugerindo
    que estamos na zona de saturacao. Esta etapa testa N=10000 para
    confirmar se a saturacao se mantem ou se o ganho marginal ainda
    e perceptivel.

Grid:
    N = 10000
    K = 5 (confirmado vencedor)

Tempo estimado:
    ~85 min (proporcional ao tempo de N=8000 = 68 min)

Uso:
    python3 sintonia_v5.py
"""
import csv
import json
import os
import time
import numpy as np

from pontos import preparar_problema
from ga import ga_caixeiro


N = 10000
K = 5
N_RODADAS = 30
MAX_GEN = 1000
P_MUT = 0.01
N_ELITE = 0
CSV_PATH = "CaixeiroGruposGA.csv"


def rodar(seed_base=20251211):
    cidades, grupos, D, n_cidades = preparar_problema(CSV_PATH)
    print(f"Problema: {n_cidades} cidades + origem")
    print(f"Sintonia v5 (saturacao): N={N}, K={K}")
    print(f"{N_RODADAS} rodadas, max_gen={MAX_GEN}, p_mut={P_MUT}\n")

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
        r = ga_caixeiro(D, n_cidades, N=N, max_gen=MAX_GEN,
                        p_mut=P_MUT, tamanho_torneio=K,
                        N_elite=N_ELITE, rng=rng)
        dt = time.time() - t_r
        custos_finais.append(r["melhor_custo"])
        hist_melhor_por_rodada.append(r["hist_melhor"])
        tempos.append(dt)
        if r["melhor_custo"] < melhor_custo:
            melhor_custo = r["melhor_custo"]
            melhor_rota = r["melhor_x"].tolist()
        print(f"  rodada {i+1:2d}/{N_RODADAS}  custo={r['melhor_custo']:7.1f}  "
              f"tempo={dt:6.1f}s  acumulado={(time.time()-t_total)/60:5.1f}min")

    arr = np.array(custos_finais)
    reg = {
        "N": N, "K": K, "rodadas": N_RODADAS, "max_gen": MAX_GEN,
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
    with open("resultados/sintonia_v5.json", "w") as fh:
        json.dump({"configs": [reg], "melhor": reg}, fh, indent=2)
    with open("resultados/sintonia_v5.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["N", "K", "media", "desvio", "minimo", "maximo",
                    "mediana", "tempo_segundos"])
        w.writerow([N, K,
                    f"{arr.mean():.2f}", f"{arr.std():.2f}",
                    f"{arr.min():.2f}", f"{arr.max():.2f}",
                    f"{np.median(arr):.2f}", f"{time.time()-t_total:.1f}"])

    print(f"\n=== Resumo ===")
    print(f"  Media:   {arr.mean():.1f}")
    print(f"  Desvio:  {arr.std():.1f}")
    print(f"  Minimo:  {arr.min():.1f}")
    print(f"  Maximo:  {arr.max():.1f}")
    print(f"  Mediana: {np.median(arr):.1f}")
    print(f"  Tempo total: {(time.time()-t_total)/60:.1f} min")


if __name__ == "__main__":
    rodar()
