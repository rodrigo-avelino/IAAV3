"""
Busca pelas 92 soluções únicas do problema das 8 rainhas usando a melhor
configuração de têmpera obtida na sintonia.

Análise de custo computacional reportada:
    - Número total de execuções da têmpera (lançamentos)
    - Número de iterações Metropolis acumuladas
    - Tempo de parede
    - Distribuição de iterações por execução bem-sucedida
    - Número médio de execuções para cada nova solução única descoberta
    - Histograma de tempo (em # execuções) entre descobertas consecutivas
"""
import json
import os
import time
import numpy as np

from tempera import tempera_simulada


T0 = 100.0
ALPHA = 0.9
MAX_IT = 5000
MAX_EXECUCOES = 50000  # salvaguarda para não loopar pra sempre


def buscar_92(seed=20251201):
    rng = np.random.default_rng(seed)

    solucoes_unicas = {}  # tupla -> primeira execução em que apareceu
    historico_execucoes_por_nova = []  # iterações entre cada nova solução
    iter_por_execucao = []
    total_iter = 0
    sucessos = 0

    print(f"Busca pelas 92 soluções únicas com T0={T0}, alpha={ALPHA}.\n")

    t0 = time.time()
    ultima_descoberta_em = 0

    for execucao in range(1, MAX_EXECUCOES + 1):
        r = tempera_simulada(T0=T0, alpha=ALPHA, max_it=MAX_IT, rng=rng)
        total_iter += r["iteracoes"]

        if r["sucesso"]:
            sucessos += 1
            iter_por_execucao.append(r["iteracoes"])
            chave = tuple(int(v) for v in r["x_best"])
            if chave not in solucoes_unicas:
                solucoes_unicas[chave] = execucao
                gap = execucao - ultima_descoberta_em
                historico_execucoes_por_nova.append(gap)
                ultima_descoberta_em = execucao
                n = len(solucoes_unicas)
                print(f"  [{n:02d}/92] exec={execucao:6d}  gap={gap:5d}  "
                      f"sol={chave}")
                if n >= 92:
                    break

    tempo_total = time.time() - t0
    taxa_sucesso = sucessos / execucao

    relatorio = {
        "T0": T0,
        "alpha": ALPHA,
        "max_it": MAX_IT,
        "solucoes_unicas_encontradas": len(solucoes_unicas),
        "execucoes_totais": execucao,
        "execucoes_com_sucesso": sucessos,
        "taxa_sucesso_global": taxa_sucesso,
        "iteracoes_metropolis_totais": int(total_iter),
        "tempo_segundos": tempo_total,
        "media_iter_por_execucao_bem_sucedida": float(np.mean(iter_por_execucao)),
        "mediana_iter_por_execucao_bem_sucedida": float(np.median(iter_por_execucao)),
        "gap_medio_entre_novas_solucoes": float(np.mean(historico_execucoes_por_nova)),
        "gap_max_entre_novas_solucoes": int(max(historico_execucoes_por_nova)),
        "gaps_descobertas": historico_execucoes_por_nova,
        "iter_por_execucao_bem_sucedida": iter_por_execucao,
        "solucoes": list(solucoes_unicas.keys()),
    }

    print(f"\n=== Relatório ===")
    print(f"  Soluções únicas encontradas       : {len(solucoes_unicas)}")
    print(f"  Execuções totais (lançamentos)    : {execucao}")
    print(f"  Execuções com sucesso             : {sucessos} ({taxa_sucesso*100:.1f}%)")
    print(f"  Iterações Metropolis acumuladas   : {total_iter:,}")
    print(f"  Tempo total                       : {tempo_total:.2f} s")
    print(f"  Média de iter por execução OK     : {np.mean(iter_por_execucao):.1f}")
    print(f"  Mediana de iter por execução OK   : {np.median(iter_por_execucao):.0f}")
    print(f"  Gap médio entre descobertas       : {np.mean(historico_execucoes_por_nova):.2f} execuções")
    print(f"  Maior gap entre descobertas       : {max(historico_execucoes_por_nova)} execuções")

    os.makedirs("resultados", exist_ok=True)
    with open("resultados/busca_92.json", "w") as fh:
        json.dump(relatorio, fh, indent=2)

    return relatorio


if __name__ == "__main__":
    buscar_92()
