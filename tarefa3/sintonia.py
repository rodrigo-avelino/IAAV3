"""
Sintonia de hiperparametros do GA para o Caixeiro Viajante 3D.

Hiperparametros sintonizados:
    N (tamanho da populacao): {50, 100, 200}
    K (tamanho do torneio):   {2, 3, 5}

Hiperparametros fixos pelo enunciado:
    p_mut    = 0.01  (item 5: "mutacao com probabilidade 1%")
    max_gen  = 1000  (item 5: definir maximo de geracoes)
    operador de selecao: torneio
    operador de crossover: 2 pontos com reparo (OX)
    operador de mutacao: swap

Elitismo:
    Fixado em 0 nesta etapa (baseline). O efeito do elitismo e analisado
    em etapa subsequente, com os hiperparametros ja sintonizados.

Criterio de escolha:
    Menor custo medio final entre as 30 rodadas. Em caso de empate, menor
    desvio-padrao (mais consistente).

Uso:
    python3 sintonia.py            # roda todas as 9 configs (~17 min)
    python3 sintonia.py 50         # roda apenas N=50 com K=2,3,5 (~2.5 min)
    python3 sintonia.py 100        # roda apenas N=100 (~5 min)
    python3 sintonia.py 200        # roda apenas N=200 (~10 min)
    python3 sintonia.py agregar    # junta os parciais em sintonia_completa.json
"""
import csv
import json
import os
import sys
import time
import numpy as np

from pontos import preparar_problema
from ga import ga_caixeiro


N_VALORES = [50, 100, 200]
K_VALORES = [2, 3, 5]
N_RODADAS = 30
MAX_GEN = 1000
P_MUT = 0.01
N_ELITE = 0
CSV_PATH = "CaixeiroGruposGA.csv"  # mesma pasta do script


def _rodar_uma_config(N, K, D, n_cidades, rng_global):
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


def rodar_sintonia(filtro_N=None, seed_base=20251201):
    cidades, grupos, D, n_cidades = preparar_problema(CSV_PATH)
    n_valores = N_VALORES if filtro_N is None else [filtro_N]
    print(f"Problema: {n_cidades} cidades + origem")
    print(f"Configs: {len(n_valores)} x {len(K_VALORES)} = "
          f"{len(n_valores) * len(K_VALORES)}, "
          f"{N_RODADAS} rodadas, max_gen={MAX_GEN}, p_mut={P_MUT}\n")

    # RNG global determinista (avanca igual em todos os modos, garantindo
    # que rodar por partes da o mesmo resultado de rodar tudo de uma vez)
    rng_global = np.random.default_rng(seed_base)
    if filtro_N is not None:
        for N_skip in N_VALORES:
            if N_skip == filtro_N:
                break
            for _ in K_VALORES:
                for _ in range(N_RODADAS):
                    rng_global.integers(0, 2**31 - 1)

    resultados = []
    t_total = time.time()
    for N in n_valores:
        for K in K_VALORES:
            r = _rodar_uma_config(N, K, D, n_cidades, rng_global)
            resultados.append(r)
            print(f"  N={N:3d}  K={K}  media={r['media']:7.1f}  "
                  f"desvio={r['desvio']:6.1f}  min={r['minimo']:6.1f}  "
                  f"max={r['maximo']:6.1f}  tempo={r['tempo_total_segundos']:5.1f}s")

    os.makedirs("resultados", exist_ok=True)
    sufixo = f"_N{filtro_N}" if filtro_N is not None else ""
    with open(f"resultados/sintonia_completa{sufixo}.json", "w") as fh:
        json.dump({"configs": resultados}, fh, indent=2)
    print(f"\nTempo parcial: {(time.time() - t_total)/60:.1f} min")

    if filtro_N is None:
        _salvar_resumo(resultados)


def _salvar_resumo(resultados):
    melhor = min(resultados, key=lambda r: (r["media"], r["desvio"]))
    print(f"MELHOR CONFIG: N={melhor['N']}, K={melhor['K']}  "
          f"(media={melhor['media']:.1f}, desvio={melhor['desvio']:.1f})")
    with open("resultados/sintonia_completa.json", "w") as fh:
        json.dump({"configs": resultados, "melhor": melhor}, fh, indent=2)
    with open("resultados/sintonia_resumo.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["N", "K", "media", "desvio", "minimo", "maximo",
                    "mediana", "tempo_segundos"])
        for r in resultados:
            w.writerow([r["N"], r["K"],
                        f"{r['media']:.2f}", f"{r['desvio']:.2f}",
                        f"{r['minimo']:.2f}", f"{r['maximo']:.2f}",
                        f"{r['mediana']:.2f}", f"{r['tempo_total_segundos']:.1f}"])


def agregar_parciais():
    """Combina os arquivos sintonia_completa_NXX.json em sintonia_completa.json."""
    todos = []
    for N in N_VALORES:
        path = f"resultados/sintonia_completa_N{N}.json"
        if not os.path.exists(path):
            print(f"AVISO: parcial faltando: {path}")
            continue
        with open(path) as fh:
            todos.extend(json.load(fh)["configs"])
    if len(todos) < len(N_VALORES) * len(K_VALORES):
        print(f"AVISO: agregando apenas {len(todos)} de "
              f"{len(N_VALORES)*len(K_VALORES)} configs esperadas")
    _salvar_resumo(todos)
    print(f"Agregado {len(todos)} configs.")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        if sys.argv[1] == "agregar":
            agregar_parciais()
        else:
            rodar_sintonia(filtro_N=int(sys.argv[1]))
    else:
        rodar_sintonia()
