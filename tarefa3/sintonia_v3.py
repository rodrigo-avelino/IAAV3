"""
Sintonia v3 do GA caixeiro viajante: confirmacao do joelho da curva.

Motivacao:
    A sintonia expandida v1 mostrou ganho marginal ainda decrescente em N
    (500->1000: -9%; 1000->2000: -8%) e descartou definitivamente K=15.
    Esta etapa testa N=3000 e N=5000 com K=5 fixo para identificar onde
    o ganho marginal torna-se computacionalmente irrelevante.

Grid:
    N in {3000, 5000}
    K = 5 (confirmado vencedor)

Tempo estimado (validado em 1 execucao):
    N=3000: ~140s/exec x 30 rodadas = ~70 min
    N=5000: ~234s/exec x 30 rodadas = ~117 min
    Total: ~187 min (~3h)

Uso:
    python3 sintonia_v3.py            # roda tudo
    python3 sintonia_v3.py 3000       # apenas N=3000
    python3 sintonia_v3.py 5000       # apenas N=5000
    python3 sintonia_v3.py agregar    # junta os parciais
"""
import csv
import json
import os
import sys
import time
import numpy as np

from pontos import preparar_problema
from ga import ga_caixeiro


N_VALORES = [3000, 5000]
K = 5
N_RODADAS = 30
MAX_GEN = 1000
P_MUT = 0.01
N_ELITE = 0
CSV_PATH = "CaixeiroGruposGA.csv"


def _rodar_uma_config(N, D, n_cidades, rng_global):
    t_cfg = time.time()
    custos_finais = []
    hist_melhor_por_rodada = []
    melhor_rota_cfg = None
    melhor_custo_cfg = np.inf

    for _ in range(N_RODADAS):
        seed_rodada = int(rng_global.integers(0, 2**31 - 1))
        rng = np.random.default_rng(seed_rodada)
        r = ga_caixeiro(D, n_cidades, N=N, max_gen=MAX_GEN,
                        p_mut=P_MUT, tamanho_torneio=K,
                        N_elite=N_ELITE, rng=rng)
        custos_finais.append(r["melhor_custo"])
        hist_melhor_por_rodada.append(r["hist_melhor"])
        if r["melhor_custo"] < melhor_custo_cfg:
            melhor_custo_cfg = r["melhor_custo"]
            melhor_rota_cfg = r["melhor_x"].tolist()

    arr = np.array(custos_finais)
    return {
        "N": N, "K": K, "rodadas": N_RODADAS, "max_gen": MAX_GEN,
        "custos_finais": custos_finais,
        "media": float(arr.mean()), "desvio": float(arr.std()),
        "minimo": float(arr.min()), "maximo": float(arr.max()),
        "mediana": float(np.median(arr)),
        "tempo_total_segundos": time.time() - t_cfg,
        "melhor_rota": melhor_rota_cfg,
        "hist_melhor_por_rodada": [list(h) for h in hist_melhor_por_rodada],
    }


def rodar(filtro_N=None, seed_base=20251209):
    cidades, grupos, D, n_cidades = preparar_problema(CSV_PATH)
    n_valores = N_VALORES if filtro_N is None else [filtro_N]
    print(f"Problema: {n_cidades} cidades + origem")
    print(f"Sintonia v3 (joelho): N in {n_valores}, K={K}")
    print(f"{N_RODADAS} rodadas, max_gen={MAX_GEN}, p_mut={P_MUT}\n")

    rng_global = np.random.default_rng(seed_base)
    if filtro_N is not None:
        for N_skip in N_VALORES:
            if N_skip == filtro_N:
                break
            for _ in range(N_RODADAS):
                rng_global.integers(0, 2**31 - 1)

    resultados = []
    t_total = time.time()
    for N in n_valores:
        r = _rodar_uma_config(N, D, n_cidades, rng_global)
        resultados.append(r)
        print(f"  N={N:4d}  K={K}  media={r['media']:7.1f}  "
              f"desvio={r['desvio']:6.1f}  min={r['minimo']:6.1f}  "
              f"max={r['maximo']:6.1f}  tempo={r['tempo_total_segundos']:6.1f}s")

    os.makedirs("resultados", exist_ok=True)
    sufixo = f"_N{filtro_N}" if filtro_N is not None else ""
    with open(f"resultados/sintonia_v3{sufixo}.json", "w") as fh:
        json.dump({"configs": resultados}, fh, indent=2)
    print(f"\nTempo parcial: {(time.time() - t_total)/60:.1f} min")

    if filtro_N is None:
        _salvar_resumo(resultados)


def _salvar_resumo(resultados):
    melhor = min(resultados, key=lambda r: (r["media"], r["desvio"]))
    print(f"MELHOR CONFIG: N={melhor['N']}, K={melhor['K']}  "
          f"(media={melhor['media']:.1f}, desvio={melhor['desvio']:.1f})")
    with open("resultados/sintonia_v3.json", "w") as fh:
        json.dump({"configs": resultados, "melhor": melhor}, fh, indent=2)
    with open("resultados/sintonia_v3.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["N", "K", "media", "desvio", "minimo", "maximo",
                    "mediana", "tempo_segundos"])
        for r in resultados:
            w.writerow([r["N"], r["K"],
                        f"{r['media']:.2f}", f"{r['desvio']:.2f}",
                        f"{r['minimo']:.2f}", f"{r['maximo']:.2f}",
                        f"{r['mediana']:.2f}", f"{r['tempo_total_segundos']:.1f}"])


def agregar_parciais():
    todos = []
    for N in N_VALORES:
        path = f"resultados/sintonia_v3_N{N}.json"
        if not os.path.exists(path):
            print(f"AVISO: parcial faltando: {path}")
            continue
        with open(path) as fh:
            todos.extend(json.load(fh)["configs"])
    if len(todos) < len(N_VALORES):
        print(f"AVISO: agregando {len(todos)} de {len(N_VALORES)} configs")
    _salvar_resumo(todos)
    print(f"Agregado {len(todos)} configs.")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        if sys.argv[1] == "agregar":
            agregar_parciais()
        else:
            rodar(filtro_N=int(sys.argv[1]))
    else:
        rodar()
