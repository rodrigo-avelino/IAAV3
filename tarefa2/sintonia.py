"""
Sintonia de hiperparâmetros (T0, alpha) para o problema das 8 Rainhas.

Grid:
    T0    in {1, 5, 10, 50, 100, 500}
    alpha in {0.80, 0.85, 0.90, 0.95, 0.99}

Para cada combinação, executa N_RODADAS independentes (com seeds distintas)
até atingir f=28 ou max_it. Métricas reportadas:
    - taxa de sucesso (% rodadas que atingiram f=28)
    - mediana de iterações entre as rodadas bem-sucedidas
    - média e desvio de iterações entre as rodadas bem-sucedidas

A "melhor" configuração é aquela que maximiza a taxa de sucesso; em caso de
empate, a que minimiza a mediana de iterações (custo computacional).
"""
import json
import os
import numpy as np

from tempera import tempera_simulada


T0_LISTA = [1.0, 5.0, 10.0, 50.0, 100.0, 500.0]
ALPHA_LISTA = [0.80, 0.85, 0.90, 0.95, 0.99]
N_RODADAS = 100
MAX_IT = 5000


def rodar_grid(seed=20251201):
    rng = np.random.default_rng(seed)

    mapa_sucesso = np.zeros((len(T0_LISTA), len(ALPHA_LISTA)))
    mapa_iter_mediana = np.zeros((len(T0_LISTA), len(ALPHA_LISTA)))
    mapa_iter_media = np.zeros((len(T0_LISTA), len(ALPHA_LISTA)))
    detalhes = []

    print(f"Grid {len(T0_LISTA)}x{len(ALPHA_LISTA)} = {len(T0_LISTA)*len(ALPHA_LISTA)} configs, "
          f"{N_RODADAS} rodadas/config.\n")

    for i, T0 in enumerate(T0_LISTA):
        for j, alpha in enumerate(ALPHA_LISTA):
            sucessos = 0
            iter_sucesso = []
            for _ in range(N_RODADAS):
                r = tempera_simulada(T0=T0, alpha=alpha, max_it=MAX_IT, rng=rng)
                if r["sucesso"]:
                    sucessos += 1
                    iter_sucesso.append(r["iteracoes"])

            taxa = sucessos / N_RODADAS
            mapa_sucesso[i, j] = taxa * 100
            if iter_sucesso:
                mapa_iter_mediana[i, j] = float(np.median(iter_sucesso))
                mapa_iter_media[i, j] = float(np.mean(iter_sucesso))
            else:
                mapa_iter_mediana[i, j] = MAX_IT
                mapa_iter_media[i, j] = MAX_IT

            detalhes.append({
                "T0": T0,
                "alpha": alpha,
                "taxa_sucesso": float(taxa),
                "mediana_iter": float(mapa_iter_mediana[i, j]),
                "media_iter": float(mapa_iter_media[i, j]),
                "n_sucessos": int(sucessos),
            })
            print(f"  T0={T0:6.1f}  alpha={alpha:.2f}  taxa={taxa*100:5.1f}%  "
                  f"mediana_it={mapa_iter_mediana[i, j]:7.1f}  media_it={mapa_iter_media[i, j]:7.1f}")

    # Escolha: máxima taxa, desempate por menor mediana de iterações
    melhor = max(detalhes,
                 key=lambda d: (d["taxa_sucesso"], -d["mediana_iter"]))

    print(f"\nMELHOR CONFIGURAÇÃO: T0={melhor['T0']}, alpha={melhor['alpha']}  "
          f"(taxa={melhor['taxa_sucesso']*100:.1f}%, mediana_it={melhor['mediana_iter']:.0f})")

    os.makedirs("resultados", exist_ok=True)
    np.savez("resultados/sintonia_grid.npz",
             T0_lista=np.array(T0_LISTA),
             alpha_lista=np.array(ALPHA_LISTA),
             mapa_sucesso=mapa_sucesso,
             mapa_iter_mediana=mapa_iter_mediana,
             mapa_iter_media=mapa_iter_media)
    with open("resultados/sintonia_detalhes.json", "w") as fh:
        json.dump({"detalhes": detalhes, "melhor": melhor}, fh, indent=2)

    return melhor


if __name__ == "__main__":
    melhor = rodar_grid()
