"""
Experimentos finais: 100 rodadas com os hiperparâmetros sintonizados.

Para cada função e algoritmo registra:
- 100 valores de f_best (um por rodada)
- Coordenada da melhor solução de cada rodada
- Histórico completo da melhor rodada e da pior rodada (para curvas e trajetórias)
- Estatísticas: média, desvio-padrão, mínimo, máximo, moda e taxa de convergência

A moda é calculada com binning adaptativo: o range de valores observados é
dividido em 50 intervalos e o bin mais frequente é retornado como (centro do
intervalo, contagem, percentual). Isso é mais robusto que arredondar a um
número fixo de casas decimais (que mistura bins de tamanho fixo em escalas
muito diferentes entre funções).
"""
import csv
import json
import os
import numpy as np

from funcoes import CONFIG, convergiu
from algoritmos import hill_climbing, local_random_search, global_random_search


N_RODADAS = 100
MAX_IT = 1000
T_PATIENCE = 100


def calcular_moda(valores, n_bins=50):
    """Moda por binning adaptativo no range observado."""
    valores = np.asarray(valores, dtype=float)
    if valores.std() < 1e-12:
        # Todos os valores praticamente iguais
        return {"centro": float(valores.mean()),
                "contagem": int(len(valores)),
                "percentual": 100.0,
                "largura_bin": 0.0}
    hist, edges = np.histogram(valores, bins=n_bins)
    idx = int(np.argmax(hist))
    centro = (edges[idx] + edges[idx + 1]) / 2.0
    return {
        "centro": float(centro),
        "contagem": int(hist[idx]),
        "percentual": float(100.0 * hist[idx] / len(valores)),
        "largura_bin": float(edges[idx + 1] - edges[idx]),
    }


def rodar_experimentos(seed=20251201):
    with open("resultados/parametros_sintonizados.json") as fh:
        params = json.load(fh)

    rng = np.random.default_rng(seed)
    relatorio = {}

    for nome, cfg in CONFIG.items():
        f, bounds, modo = cfg["funcao"], cfg["bounds"], cfg["modo"]
        eps = params[nome]["epsilon_hc"]
        sig = params[nome]["sigma_lrs"]

        print(f"\n=== {nome.upper()} ({modo}) ===")
        print(f"  HP: epsilon={eps:.4f}  sigma={sig:.2f}  otimo={cfg['otimo']:.6f}")

        dados = {alg: {"f": [], "x": [], "hists": []} for alg in ("HC", "LRS", "GRS")}

        for _ in range(N_RODADAS):
            r = hill_climbing(f, bounds, eps, modo, MAX_IT, T_PATIENCE, rng)
            dados["HC"]["f"].append(r["f_best"])
            dados["HC"]["x"].append(r["x_best"])
            dados["HC"]["hists"].append(r)

            r = local_random_search(f, bounds, sig, modo, MAX_IT, T_PATIENCE, rng)
            dados["LRS"]["f"].append(r["f_best"])
            dados["LRS"]["x"].append(r["x_best"])
            dados["LRS"]["hists"].append(r)

            r = global_random_search(f, bounds, modo, MAX_IT, rng=rng)
            dados["GRS"]["f"].append(r["f_best"])
            dados["GRS"]["x"].append(r["x_best"])
            dados["GRS"]["hists"].append(r)

        relatorio[nome] = {}
        for alg, d in dados.items():
            arr = np.array(d["f"])
            n_conv = sum(convergiu(v, nome) for v in d["f"])
            moda = calcular_moda(arr)
            est = {
                "media": float(arr.mean()),
                "desvio": float(arr.std()),
                "minimo": float(arr.min()),
                "maximo": float(arr.max()),
                "moda_centro": moda["centro"],
                "moda_contagem": moda["contagem"],
                "moda_percentual": moda["percentual"],
                "moda_largura_bin": moda["largura_bin"],
                "taxa_convergencia": float(n_conv / N_RODADAS),
            }
            relatorio[nome][alg] = est
            relatorio[nome][alg + "_dados"] = {
                "f": [float(v) for v in d["f"]],
                "x": [list(map(float, x)) for x in d["x"]],
            }

            print(f"  {alg:3s}: media={est['media']:10.5f} | desvio={est['desvio']:10.5f} | "
                  f"min={est['minimo']:10.5f} | max={est['maximo']:10.5f} | "
                  f"moda={est['moda_centro']:8.4f} ({est['moda_contagem']}x) | "
                  f"conv={est['taxa_convergencia']*100:5.1f}%")

        # Guardar histórias para os gráficos
        relatorio[nome]["_hists"] = {alg: dados[alg]["hists"] for alg in ("HC", "LRS", "GRS")}

    return relatorio


def salvar_tabela_csv(relatorio):
    """Salva tabela consolidada de estatísticas (formato pronto para o relatório)."""
    path = "resultados/tabela_estatisticas.csv"
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["Funcao", "Modo", "Algoritmo", "Otimo", "Media", "Desvio",
                    "Minimo", "Maximo", "Moda", "Moda_Contagem", "Taxa_Convergencia"])
        for nome, info in relatorio.items():
            cfg = CONFIG[nome]
            for alg in ("HC", "LRS", "GRS"):
                est = info[alg]
                w.writerow([
                    nome.upper(),
                    "Min" if cfg["modo"] == "min" else "Max",
                    alg,
                    f"{cfg['otimo']:.6f}",
                    f"{est['media']:.5f}",
                    f"{est['desvio']:.5f}",
                    f"{est['minimo']:.5f}",
                    f"{est['maximo']:.5f}",
                    f"{est['moda_centro']:.5f}",
                    est["moda_contagem"],
                    f"{est['taxa_convergencia']:.3f}",
                ])
    print(f"\nTabela salva em {path}")


def salvar_dados_brutos(relatorio):
    """CSV bruto por função (uma linha por rodada, três colunas de algoritmos)."""
    for nome, info in relatorio.items():
        path = f"resultados/dados_brutos_{nome}.csv"
        with open(path, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["Rodada", "HC", "LRS", "GRS"])
            for i in range(N_RODADAS):
                w.writerow([i + 1,
                            f"{info['HC_dados']['f'][i]:.6f}",
                            f"{info['LRS_dados']['f'][i]:.6f}",
                            f"{info['GRS_dados']['f'][i]:.6f}"])


if __name__ == "__main__":
    os.makedirs("resultados", exist_ok=True)
    rel = rodar_experimentos()
    salvar_tabela_csv(rel)
    salvar_dados_brutos(rel)

    # Salvar o relatório completo (sem histórias inteiras para não explodir o JSON)
    sem_hists = {}
    for nome, info in rel.items():
        sem_hists[nome] = {k: v for k, v in info.items() if k != "_hists"}
    with open("resultados/relatorio_completo.json", "w") as fh:
        json.dump(sem_hists, fh, indent=2, ensure_ascii=False)

    # Salvar histórias separadamente em .npz pra gráficos depois
    import pickle
    hists_para_grafico = {nome: rel[nome]["_hists"] for nome in rel}
    with open("resultados/historicos.pkl", "wb") as fh:
        pickle.dump(hists_para_grafico, fh)

    print("\nExperimentos finalizados.")
