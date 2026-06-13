"""
Analise do efeito do elitismo no GA Caixeiro Viajante.

Hiperparametros sintonizados (etapa anterior):
    N = 200, K = 5, p_mut = 0.01, max_gen = 1000

Faixa de N_elite testada:
    0  -> baseline sem elitismo
    1  -> Goldberg 1989 (apenas o melhor preservado)
    5  -> elitismo moderado (2.5% da populacao)
    10 -> elitismo padrao (5% da populacao, regra de bolso Eiben & Smith)
    20 -> elitismo alto (10% da populacao, limite antes de perder diversidade)

Cada configuracao executa 30 rodadas independentes; relata media, desvio,
minimo, maximo e mediana do custo final. Salva tambem as curvas de
convergencia (melhor por geracao em cada rodada) para gerar a figura
comparativa.

Uso:
    python3 elitismo.py            # roda todas as 5 configs (~40-45 min)
    python3 elitismo.py 0          # roda apenas N_elite=0
    python3 elitismo.py 1          # roda apenas N_elite=1
    python3 elitismo.py 5          # roda apenas N_elite=5
    python3 elitismo.py 10         # roda apenas N_elite=10
    python3 elitismo.py 20         # roda apenas N_elite=20
    python3 elitismo.py agregar    # junta os parciais
"""
import csv
import json
import os
import sys
import time
import numpy as np

from pontos import preparar_problema
from ga import ga_caixeiro


# Hiperparametros (vencedores da sintonia anterior)
N_POP = 200
K_TORNEIO = 5
P_MUT = 0.01
MAX_GEN = 1000

# Faixa de elitismo a analisar
N_ELITE_VALORES = [0, 1, 5, 10, 20]
N_RODADAS = 30
CSV_PATH = "CaixeiroGruposGA.csv"


def _rodar_uma_config(N_elite, D, n_cidades, rng_global):
    t_cfg = time.time()
    custos_finais = []
    hist_melhor_por_rodada = []
    melhor_rota_cfg = None
    melhor_custo_cfg = np.inf

    for _ in range(N_RODADAS):
        seed_rodada = int(rng_global.integers(0, 2**31 - 1))
        rng = np.random.default_rng(seed_rodada)
        r = ga_caixeiro(D, n_cidades, N=N_POP, max_gen=MAX_GEN,
                        p_mut=P_MUT, tamanho_torneio=K_TORNEIO,
                        N_elite=N_elite, rng=rng)
        custos_finais.append(r["melhor_custo"])
        hist_melhor_por_rodada.append(r["hist_melhor"])
        if r["melhor_custo"] < melhor_custo_cfg:
            melhor_custo_cfg = r["melhor_custo"]
            melhor_rota_cfg = r["melhor_x"].tolist()

    arr = np.array(custos_finais)
    return {
        "N_elite": N_elite, "N": N_POP, "K": K_TORNEIO,
        "rodadas": N_RODADAS, "max_gen": MAX_GEN,
        "custos_finais": custos_finais,
        "media": float(arr.mean()), "desvio": float(arr.std()),
        "minimo": float(arr.min()), "maximo": float(arr.max()),
        "mediana": float(np.median(arr)),
        "tempo_total_segundos": time.time() - t_cfg,
        "melhor_rota": melhor_rota_cfg,
        "hist_melhor_por_rodada": [list(h) for h in hist_melhor_por_rodada],
    }


def rodar_analise(filtro_Ne=None, seed_base=20251202):
    cidades, grupos, D, n_cidades = preparar_problema(CSV_PATH)
    valores = N_ELITE_VALORES if filtro_Ne is None else [filtro_Ne]
    print(f"Problema: {n_cidades} cidades + origem")
    print(f"Configs fixas: N={N_POP}, K={K_TORNEIO}, p_mut={P_MUT}, max_gen={MAX_GEN}")
    print(f"Variando N_elite em {valores}, {N_RODADAS} rodadas cada\n")

    # RNG global determinista: mesmo resultado rodando tudo ou por partes
    rng_global = np.random.default_rng(seed_base)
    if filtro_Ne is not None:
        for Ne_skip in N_ELITE_VALORES:
            if Ne_skip == filtro_Ne:
                break
            for _ in range(N_RODADAS):
                rng_global.integers(0, 2**31 - 1)

    resultados = []
    t_total = time.time()
    for Ne in valores:
        r = _rodar_uma_config(Ne, D, n_cidades, rng_global)
        resultados.append(r)
        print(f"  N_elite={Ne:2d}  media={r['media']:7.1f}  "
              f"desvio={r['desvio']:6.1f}  min={r['minimo']:6.1f}  "
              f"max={r['maximo']:6.1f}  tempo={r['tempo_total_segundos']:5.1f}s")

    os.makedirs("resultados", exist_ok=True)
    sufixo = f"_Ne{filtro_Ne}" if filtro_Ne is not None else ""
    with open(f"resultados/elitismo_completa{sufixo}.json", "w") as fh:
        json.dump({"configs": resultados}, fh, indent=2)
    print(f"\nTempo parcial: {(time.time() - t_total)/60:.1f} min")

    if filtro_Ne is None:
        _salvar_resumo(resultados)


def _salvar_resumo(resultados):
    melhor = min(resultados, key=lambda r: (r["media"], r["desvio"]))
    print(f"MELHOR N_elite = {melhor['N_elite']}  "
          f"(media={melhor['media']:.1f}, desvio={melhor['desvio']:.1f})")
    with open("resultados/elitismo_completa.json", "w") as fh:
        json.dump({"configs": resultados, "melhor": melhor}, fh, indent=2)
    with open("resultados/elitismo_resumo.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["N_elite", "media", "desvio", "minimo", "maximo",
                    "mediana", "tempo_segundos"])
        for r in resultados:
            w.writerow([r["N_elite"],
                        f"{r['media']:.2f}", f"{r['desvio']:.2f}",
                        f"{r['minimo']:.2f}", f"{r['maximo']:.2f}",
                        f"{r['mediana']:.2f}", f"{r['tempo_total_segundos']:.1f}"])


def agregar_parciais():
    todos = []
    for Ne in N_ELITE_VALORES:
        path = f"resultados/elitismo_completa_Ne{Ne}.json"
        if not os.path.exists(path):
            print(f"AVISO: parcial faltando: {path}")
            continue
        with open(path) as fh:
            todos.extend(json.load(fh)["configs"])
    if len(todos) < len(N_ELITE_VALORES):
        print(f"AVISO: agregando {len(todos)} de {len(N_ELITE_VALORES)} configs")
    _salvar_resumo(todos)
    print(f"Agregado {len(todos)} configs.")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        if sys.argv[1] == "agregar":
            agregar_parciais()
        else:
            rodar_analise(filtro_Ne=int(sys.argv[1]))
    else:
        rodar_analise()
