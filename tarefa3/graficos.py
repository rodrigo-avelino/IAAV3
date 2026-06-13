"""
Figuras da sintonia do GA Caixeiro Viajante.

1. Heatmap N x K com m\u00e9dia e desvio do custo final.
2. Boxplots por configura\u00e7\u00e3o (9 caixas lado a lado).
3. Curvas de converg\u00eancia por configura\u00e7\u00e3o (3x3 grid, envelope min-max).
4. Melhor rota 3D encontrada na sintonia (qualquer config).
"""
import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pontos import preparar_problema


def carregar():
    with open("resultados/sintonia_completa.json") as fh:
        return json.load(fh)


def carregar_elitismo():
    with open("resultados/elitismo_completa.json") as fh:
        return json.load(fh)


def carregar_final():
    with open("resultados/final_5000ger.json") as fh:
        return json.load(fh)


def carregar_expandida():
    path = "resultados/sintonia_expandida.json"
    if not os.path.exists(path):
        return None
    with open(path) as fh:
        return json.load(fh)


def carregar_v3():
    path = "resultados/sintonia_v3.json"
    if not os.path.exists(path):
        return None
    with open(path) as fh:
        return json.load(fh)


def carregar_v4():
    path = "resultados/sintonia_v4.json"
    if not os.path.exists(path):
        return None
    with open(path) as fh:
        return json.load(fh)


def plot_heatmap(data):
    configs = data["configs"]
    Ns = sorted(set(c["N"] for c in configs))
    Ks = sorted(set(c["K"] for c in configs))
    media = np.zeros((len(Ns), len(Ks)))
    desvio = np.zeros((len(Ns), len(Ks)))
    for c in configs:
        i, j = Ns.index(c["N"]), Ks.index(c["K"])
        media[i, j] = c["media"]
        desvio[i, j] = c["desvio"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    im1 = ax1.imshow(media, cmap="RdYlGn_r", aspect="auto")
    ax1.set_xticks(range(len(Ks))); ax1.set_xticklabels(Ks)
    ax1.set_yticks(range(len(Ns))); ax1.set_yticklabels(Ns)
    ax1.set_xlabel("K (torneio)"); ax1.set_ylabel("N (popula\u00e7\u00e3o)")
    ax1.set_title("Custo m\u00e9dio (30 rodadas)")
    for i in range(len(Ns)):
        for j in range(len(Ks)):
            v = media[i, j]
            cor = "white" if v > media.mean() else "black"
            ax1.text(j, i, f"{v:.0f}", ha="center", va="center",
                     color=cor, fontweight="bold", fontsize=10)
    fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)

    im2 = ax2.imshow(desvio, cmap="YlOrRd", aspect="auto")
    ax2.set_xticks(range(len(Ks))); ax2.set_xticklabels(Ks)
    ax2.set_yticks(range(len(Ns))); ax2.set_yticklabels(Ns)
    ax2.set_xlabel("K (torneio)"); ax2.set_ylabel("N (popula\u00e7\u00e3o)")
    ax2.set_title("Desvio-padr\u00e3o")
    for i in range(len(Ns)):
        for j in range(len(Ks)):
            v = desvio[i, j]
            cor = "white" if v > 0.6 * desvio.max() else "black"
            ax2.text(j, i, f"{v:.0f}", ha="center", va="center",
                     color=cor, fontweight="bold", fontsize=10)
    fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.savefig("resultados/heatmap_sintonia.png", dpi=140)
    plt.close()


def plot_boxplots(data):
    configs = sorted(data["configs"], key=lambda c: (c["N"], c["K"]))
    dados = [c["custos_finais"] for c in configs]
    labels = [f"N={c['N']}\nK={c['K']}" for c in configs]

    fig, ax = plt.subplots(figsize=(10, 4.5))
    bp = ax.boxplot(dados, tick_labels=labels, widths=0.6, patch_artist=True)

    # Destaca o melhor
    melhor_idx = min(range(len(configs)), key=lambda i: configs[i]["media"])
    for i, patch in enumerate(bp["boxes"]):
        patch.set_facecolor("#2ca02c" if i == melhor_idx else "#cccccc")
        patch.set_alpha(0.7)

    ax.set_ylabel("Custo final (dist\u00e2ncia total)")
    ax.set_title("Distribui\u00e7\u00e3o do custo final por configura\u00e7\u00e3o (30 rodadas)\n"
                 "verde = melhor config (menor m\u00e9dia)")
    ax.grid(alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig("resultados/boxplots_sintonia.png", dpi=140)
    plt.close()


def plot_convergencia_grid(data):
    configs = sorted(data["configs"], key=lambda c: (c["N"], c["K"]))
    Ns = sorted(set(c["N"] for c in configs))
    Ks = sorted(set(c["K"] for c in configs))

    fig, axes = plt.subplots(len(Ns), len(Ks), figsize=(11, 8), sharex=True, sharey=True)

    # eixo y global: min de todos os hist_melhor at\u00e9 max do hist_pior
    ylim_min = min(min(min(h) for h in c["hist_melhor_por_rodada"]) for c in configs)
    ylim_max = max(max(max(h) for h in c["hist_melhor_por_rodada"]) for c in configs)

    melhor_idx = min(range(len(configs)), key=lambda i: configs[i]["media"])

    for c in configs:
        i, j = Ns.index(c["N"]), Ks.index(c["K"])
        ax = axes[i, j]
        H = np.array(c["hist_melhor_por_rodada"])
        media = H.mean(axis=0)
        minimo = H.min(axis=0)
        maximo = H.max(axis=0)
        iters = np.arange(len(media))

        cor = "#2ca02c" if configs.index(c) == melhor_idx else "#1f77b4"
        ax.plot(iters, media, color=cor, lw=1.2)
        ax.fill_between(iters, minimo, maximo, color=cor, alpha=0.2)
        ax.set_title(f"N={c['N']}, K={c['K']}  (m\u00e9d={c['media']:.0f})",
                     fontsize=9)
        ax.grid(alpha=0.3)
        ax.set_ylim(ylim_min * 0.95, ylim_max * 1.02)

    for i in range(len(Ns)):
        axes[i, 0].set_ylabel("Custo")
    for j in range(len(Ks)):
        axes[-1, j].set_xlabel("Gera\u00e7\u00e3o")

    fig.suptitle("Converg\u00eancia por configura\u00e7\u00e3o: m\u00e9dia + envelope min\u2013max (30 rodadas)",
                 y=1.00, fontsize=11)
    plt.tight_layout()
    plt.savefig("resultados/convergencia_grid.png", dpi=140)
    plt.close()


def plot_melhor_rota(data, csv_path):
    cidades, grupos, D, n_cidades = preparar_problema(csv_path)

    # Acha a config com melhor rota (menor custo individual)
    melhor_cfg = min(data["configs"],
                     key=lambda c: c["minimo"])
    rota_idx = np.array(melhor_cfg["melhor_rota"])
    rota = np.concatenate([[0], rota_idx, [0]])
    coord = cidades[rota]
    custo = float(melhor_cfg["minimo"])

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    cores = {1: "#1f77b4", 2: "#2ca02c", 3: "#9467bd", 4: "#d62728"}
    for g in [1, 2, 3, 4]:
        mask = grupos == g
        ax.scatter(cidades[mask, 0], cidades[mask, 1], cidades[mask, 2],
                   c=cores[g], s=18, alpha=0.7, label=f"Grupo {g}")
    ax.scatter([cidades[0, 0]], [cidades[0, 1]], [cidades[0, 2]],
               c="black", s=180, marker="*", edgecolor="yellow",
               linewidth=1.2, zorder=10, label="Origem")
    ax.plot(coord[:, 0], coord[:, 1], coord[:, 2], "k-", lw=0.6, alpha=0.5)
    ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
    ax.set_title(f"Melhor rota encontrada (N={melhor_cfg['N']}, K={melhor_cfg['K']})\n"
                 f"custo = {custo:.0f}")
    ax.legend(fontsize=8, loc="upper left")
    plt.tight_layout()
    plt.savefig("resultados/melhor_rota_3d.png", dpi=140)
    plt.close()


# =========================================================================
# An\u00e1lise de elitismo (etapa posterior \u00e0 sintonia)
# =========================================================================

def plot_elitismo_boxplot(data):
    configs = sorted(data["configs"], key=lambda c: c["N_elite"])
    dados = [c["custos_finais"] for c in configs]
    labels = [f"N_elite={c['N_elite']}" for c in configs]

    fig, ax = plt.subplots(figsize=(9, 4.5))
    bp = ax.boxplot(dados, tick_labels=labels, widths=0.6, patch_artist=True)

    melhor_idx = min(range(len(configs)), key=lambda i: configs[i]["media"])
    for i, patch in enumerate(bp["boxes"]):
        patch.set_facecolor("#2ca02c" if i == melhor_idx else "#cccccc")
        patch.set_alpha(0.7)

    ax.set_ylabel("Custo final (dist\u00e2ncia total)")
    ax.set_title("Efeito do elitismo no custo final (30 rodadas)\n"
                 "verde = melhor configura\u00e7\u00e3o")
    ax.grid(alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig("resultados/elitismo_boxplot.png", dpi=140)
    plt.close()


def plot_elitismo_convergencia(data):
    configs = sorted(data["configs"], key=lambda c: c["N_elite"])
    fig, ax = plt.subplots(figsize=(9, 5))

    cmap = plt.get_cmap("viridis")
    cores = [cmap(i / max(1, len(configs) - 1)) for i in range(len(configs))]

    for cor, c in zip(cores, configs):
        H = np.array(c["hist_melhor_por_rodada"])
        media = H.mean(axis=0)
        minimo = H.min(axis=0)
        maximo = H.max(axis=0)
        iters = np.arange(len(media))
        ax.plot(iters, media, color=cor, lw=1.5,
                label=f"N_elite={c['N_elite']} (m\u00e9dia final {c['media']:.0f})")
        ax.fill_between(iters, minimo, maximo, color=cor, alpha=0.10)

    ax.set_xlabel("Gera\u00e7\u00e3o")
    ax.set_ylabel("Custo da melhor rota")
    ax.set_title("Converg\u00eancia por valor de elitismo "
                 "(N=200, K=5, p_mut=0.01, 30 rodadas)\n"
                 "linha = m\u00e9dia, faixa = envelope min\u2013max")
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("resultados/elitismo_convergencia.png", dpi=140)
    plt.close()


def plot_elitismo_barras(data):
    """Vista de auditoria: m\u00e9dia \u00b1 desvio em barras lado a lado."""
    configs = sorted(data["configs"], key=lambda c: c["N_elite"])
    Ne = [c["N_elite"] for c in configs]
    medias = [c["media"] for c in configs]
    desvios = [c["desvio"] for c in configs]
    minimos = [c["minimo"] for c in configs]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = np.arange(len(Ne))
    barras = ax.bar(x, medias, yerr=desvios, capsize=6,
                    color="#1f77b4", alpha=0.7, edgecolor="black",
                    label="m\u00e9dia \u00b1 desvio")
    ax.scatter(x, minimos, color="red", marker="v", s=70, zorder=5,
               label="m\u00ednimo observado")

    ax.set_xticks(x); ax.set_xticklabels([f"{ne}" for ne in Ne])
    ax.set_xlabel("N_elite")
    ax.set_ylabel("Custo final")
    ax.set_title("Custo m\u00e9dio \u00b1 desvio por valor de elitismo")
    ax.legend()
    ax.grid(alpha=0.3, axis="y")

    # Anota os valores
    for i, (m, d) in enumerate(zip(medias, desvios)):
        ax.text(i, m + d + 50, f"{m:.0f}", ha="center", fontsize=9)

    plt.tight_layout()
    plt.savefig("resultados/elitismo_barras.png", dpi=140)
    plt.close()


def plot_elitismo_moda_geracoes(data, percentual_tolerancia=0.50):
    """
    Para cada rodada, identifica a primeira gera\u00e7\u00e3o em que a curva
    "melhor at\u00e9 agora" atingiu um custo aceit\u00e1vel. "Aceit\u00e1vel" \u00e9
    definido como dentro de (1 + percentual_tolerancia) do menor custo
    observado em toda a an\u00e1lise (across configs e rodadas).

    Plota a moda dessas gera\u00e7\u00f5es para cada valor de N_elite e o
    histograma das gera\u00e7\u00f5es (apenas para a melhor configura\u00e7\u00e3o).
    """
    configs = sorted(data["configs"], key=lambda c: c["N_elite"])
    melhor_global = min(min(min(h) for h in c["hist_melhor_por_rodada"])
                        for c in configs)
    limiar = melhor_global * (1 + percentual_tolerancia)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))

    medianas_por_ne = []
    modas_por_ne = []
    todas_geracoes_por_ne = {}
    for c in configs:
        geracoes_aceitavel = []
        for h in c["hist_melhor_por_rodada"]:
            arr = np.asarray(h)
            idx = np.argmax(arr <= limiar)
            if arr[idx] <= limiar:
                geracoes_aceitavel.append(int(idx))
        if not geracoes_aceitavel:
            medianas_por_ne.append(float("nan"))
            modas_por_ne.append(float("nan"))
            todas_geracoes_por_ne[c["N_elite"]] = []
            continue
        medianas_por_ne.append(float(np.median(geracoes_aceitavel)))
        # Moda por binning (bins de 50 gera\u00e7\u00f5es)
        hist, edges = np.histogram(geracoes_aceitavel,
                                    bins=np.arange(0, 1050, 50))
        idx_moda = int(np.argmax(hist))
        moda = (edges[idx_moda] + edges[idx_moda + 1]) / 2.0
        modas_por_ne.append(float(moda))
        todas_geracoes_por_ne[c["N_elite"]] = geracoes_aceitavel

    Ne = [c["N_elite"] for c in configs]
    x = np.arange(len(Ne))
    ax1.bar(x - 0.2, medianas_por_ne, width=0.4, color="#1f77b4",
            alpha=0.7, edgecolor="black", label="mediana")
    ax1.bar(x + 0.2, modas_por_ne, width=0.4, color="#ff7f0e",
            alpha=0.7, edgecolor="black", label="moda (bins de 50)")
    ax1.set_xticks(x); ax1.set_xticklabels([f"{ne}" for ne in Ne])
    ax1.set_xlabel("N_elite")
    ax1.set_ylabel(f"Gera\u00e7\u00e3o para atingir custo \u2264 {limiar:.0f}")
    ax1.set_title(f"Gera\u00e7\u00f5es at\u00e9 solu\u00e7\u00e3o aceit\u00e1vel\n"
                  f"(aceit\u00e1vel = melhor global {melhor_global:.0f} + {percentual_tolerancia*100:.0f}%)")
    ax1.legend(); ax1.grid(alpha=0.3, axis="y")

    # Histograma para a melhor config (por media)
    melhor_cfg = min(configs, key=lambda c: c["media"])
    ger_melhor = todas_geracoes_por_ne[melhor_cfg["N_elite"]]
    ax2.hist(ger_melhor, bins=20, color="#2ca02c",
             alpha=0.7, edgecolor="black")
    ax2.set_xlabel("Gera\u00e7\u00e3o em que atingiu solu\u00e7\u00e3o aceit\u00e1vel")
    ax2.set_ylabel("Frequ\u00eancia (de 30 rodadas)")
    ax2.set_title(f"Histograma para N_elite={melhor_cfg['N_elite']} "
                  f"(melhor config)\n"
                  f"mediana = {np.median(ger_melhor):.0f}, "
                  f"m\u00e9dia = {np.mean(ger_melhor):.0f}")
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig("resultados/elitismo_moda_geracoes.png", dpi=140)
    plt.close()

    return {"limiar": limiar, "melhor_global": melhor_global,
            "tolerancia": percentual_tolerancia,
            "modas": dict(zip(Ne, modas_por_ne)),
            "medianas": dict(zip(Ne, medianas_por_ne))}


def plot_elitismo_melhor_rota(data, csv_path):
    cidades, grupos, _, _ = preparar_problema(csv_path)
    melhor_cfg = min(data["configs"], key=lambda c: c["minimo"])
    rota_idx = np.array(melhor_cfg["melhor_rota"])
    rota = np.concatenate([[0], rota_idx, [0]])
    coord = cidades[rota]
    custo = float(melhor_cfg["minimo"])

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    cores = {1: "#1f77b4", 2: "#2ca02c", 3: "#9467bd", 4: "#d62728"}
    for g in [1, 2, 3, 4]:
        mask = grupos == g
        ax.scatter(cidades[mask, 0], cidades[mask, 1], cidades[mask, 2],
                   c=cores[g], s=18, alpha=0.7, label=f"Grupo {g}")
    ax.scatter([cidades[0, 0]], [cidades[0, 1]], [cidades[0, 2]],
               c="black", s=180, marker="*", edgecolor="yellow",
               linewidth=1.2, zorder=10, label="Origem")
    ax.plot(coord[:, 0], coord[:, 1], coord[:, 2], "k-", lw=0.6, alpha=0.5)
    ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
    ax.set_title(f"Melhor rota da an\u00e1lise de elitismo "
                 f"(N_elite={melhor_cfg['N_elite']})\ncusto = {custo:.0f}")
    ax.legend(fontsize=8, loc="upper left")
    plt.tight_layout()
    plt.savefig("resultados/elitismo_melhor_rota_3d.png", dpi=140)
    plt.close()


# =========================================================================
# An\u00e1lise final (5000 gera\u00e7\u00f5es, N_elite=0)
# =========================================================================

def plot_final_convergencia(data, data_elitismo=None):
    """
    Curva de converg\u00eancia da execu\u00e7\u00e3o final (5000 gera\u00e7\u00f5es).
    Se data_elitismo for fornecido, sobrep\u00f5e a curva anterior (1000 ger)
    da mesma configura\u00e7\u00e3o N_elite=0 para comparar.
    """
    fig, ax = plt.subplots(figsize=(10, 5))

    H = np.array(data["hist_melhor_por_rodada"])
    media = H.mean(axis=0)
    minimo = H.min(axis=0)
    maximo = H.max(axis=0)
    iters = np.arange(len(media))
    ax.plot(iters, media, color="#1f77b4", lw=1.5,
            label=f"5000 ger (m\u00e9d. final {data['media']:.0f})")
    ax.fill_between(iters, minimo, maximo, color="#1f77b4", alpha=0.15)

    if data_elitismo is not None:
        ne0 = next(c for c in data_elitismo["configs"] if c["N_elite"] == 0)
        H0 = np.array(ne0["hist_melhor_por_rodada"])
        media0 = H0.mean(axis=0)
        iters0 = np.arange(len(media0))
        ax.plot(iters0, media0, color="#d62728", lw=1.5, linestyle="--",
                label=f"1000 ger (m\u00e9d. final {ne0['media']:.0f})")

    ax.axvline(1000, color="gray", linestyle=":", lw=1.0,
               label="limite anterior (1000)")
    ax.set_xlabel("Gera\u00e7\u00e3o")
    ax.set_ylabel("Custo da melhor rota")
    ax.set_title(f"An\u00e1lise final: N=200, K=5, N_elite=0, 5000 gera\u00e7\u00f5es "
                 f"(30 rodadas)\nlinha = m\u00e9dia, faixa = envelope min\u2013max")
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("resultados/final_convergencia.png", dpi=140)
    plt.close()


def plot_final_boxplot_comparativo(data, data_elitismo=None):
    """Boxplot comparando a execu\u00e7\u00e3o final (5000 ger) com a anterior (1000 ger)."""
    dados = [data["custos_finais"]]
    labels = [f"5000 ger\n(m\u00e9d {data['media']:.0f})"]
    cores = ["#2ca02c"]

    if data_elitismo is not None:
        ne0 = next(c for c in data_elitismo["configs"] if c["N_elite"] == 0)
        dados.insert(0, ne0["custos_finais"])
        labels.insert(0, f"1000 ger\n(m\u00e9d {ne0['media']:.0f})")
        cores.insert(0, "#cccccc")

    fig, ax = plt.subplots(figsize=(6, 4.5))
    bp = ax.boxplot(dados, tick_labels=labels, widths=0.5, patch_artist=True)
    for patch, c in zip(bp["boxes"], cores):
        patch.set_facecolor(c); patch.set_alpha(0.7)
    ax.set_ylabel("Custo final")
    ax.set_title("Compara\u00e7\u00e3o 1000 ger vs 5000 ger\n"
                 "(mesma config: N=200, K=5, N_elite=0)")
    ax.grid(alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig("resultados/final_boxplot_comparativo.png", dpi=140)
    plt.close()


def plot_final_melhor_rota(data, csv_path):
    cidades, grupos, _, _ = preparar_problema(csv_path)
    rota_idx = np.array(data["melhor_rota"])
    rota = np.concatenate([[0], rota_idx, [0]])
    coord = cidades[rota]
    custo = float(data["minimo"])

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    cores = {1: "#1f77b4", 2: "#2ca02c", 3: "#9467bd", 4: "#d62728"}
    for g in [1, 2, 3, 4]:
        mask = grupos == g
        ax.scatter(cidades[mask, 0], cidades[mask, 1], cidades[mask, 2],
                   c=cores[g], s=18, alpha=0.7, label=f"Grupo {g}")
    ax.scatter([cidades[0, 0]], [cidades[0, 1]], [cidades[0, 2]],
               c="black", s=180, marker="*", edgecolor="yellow",
               linewidth=1.2, zorder=10, label="Origem")
    ax.plot(coord[:, 0], coord[:, 1], coord[:, 2], "k-", lw=0.6, alpha=0.5)
    ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
    ax.set_title(f"Melhor rota da an\u00e1lise final "
                 f"(5000 ger, N_elite=0)\ncusto = {custo:.0f}")
    ax.legend(fontsize=8, loc="upper left")
    plt.tight_layout()
    plt.savefig("resultados/final_melhor_rota_3d.png", dpi=140)
    plt.close()


# =========================================================================
# Sintonia expandida (Tarefa 3): N x K combinado original + expandido
# =========================================================================

def plot_heatmap_expandido(data_original, data_expandido, data_v3=None, data_v4=None):
    """
    Combina o grid original {50,100,200} x {2,3,5} com o expandido
    {500,1000,2000} x {5,15} e opcionalmente com a sintonia v3
    {3000,5000} x {5} e v4 {8000} x {5} em um unico heatmap.
    """
    todos = list(data_original["configs"]) + list(data_expandido["configs"])
    if data_v3 is not None:
        todos = todos + list(data_v3["configs"])
    if data_v4 is not None:
        todos = todos + list(data_v4["configs"])
    # Remove duplicatas mantendo a primeira ocorrencia
    visto = set()
    unicos = []
    for c in todos:
        chave = (c["N"], c["K"])
        if chave not in visto:
            visto.add(chave)
            unicos.append(c)

    Ns = sorted(set(c["N"] for c in unicos))
    Ks = sorted(set(c["K"] for c in unicos))
    media = np.full((len(Ns), len(Ks)), np.nan)
    desvio = np.full((len(Ns), len(Ks)), np.nan)
    for c in unicos:
        i, j = Ns.index(c["N"]), Ks.index(c["K"])
        media[i, j] = c["media"]
        desvio[i, j] = c["desvio"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5))

    im1 = ax1.imshow(media, cmap="RdYlGn_r", aspect="auto")
    ax1.set_xticks(range(len(Ks))); ax1.set_xticklabels(Ks)
    ax1.set_yticks(range(len(Ns))); ax1.set_yticklabels(Ns)
    ax1.set_xlabel("K (torneio)"); ax1.set_ylabel("N (popula\u00e7\u00e3o)")
    ax1.set_title("Custo m\u00e9dio \u2014 grid combinado")
    for i in range(len(Ns)):
        for j in range(len(Ks)):
            v = media[i, j]
            if np.isnan(v):
                ax1.text(j, i, "\u2014", ha="center", va="center",
                         color="gray", fontsize=10)
            else:
                cor = ("white" if v > np.nanmean(media) else "black")
                ax1.text(j, i, f"{v:.0f}", ha="center", va="center",
                         color=cor, fontweight="bold", fontsize=9)
    fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)

    im2 = ax2.imshow(desvio, cmap="YlOrRd", aspect="auto")
    ax2.set_xticks(range(len(Ks))); ax2.set_xticklabels(Ks)
    ax2.set_yticks(range(len(Ns))); ax2.set_yticklabels(Ns)
    ax2.set_xlabel("K (torneio)"); ax2.set_ylabel("N (popula\u00e7\u00e3o)")
    ax2.set_title("Desvio-padr\u00e3o \u2014 grid combinado")
    vmax = np.nanmax(desvio)
    for i in range(len(Ns)):
        for j in range(len(Ks)):
            v = desvio[i, j]
            if np.isnan(v):
                ax2.text(j, i, "\u2014", ha="center", va="center",
                         color="gray", fontsize=10)
            else:
                cor = "white" if v > 0.6 * vmax else "black"
                ax2.text(j, i, f"{v:.0f}", ha="center", va="center",
                         color=cor, fontweight="bold", fontsize=9)
    fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)

    # Marca os contornos que separam original vs expandido
    fig.suptitle("Sintonia COMBINADA (original + expans\u00e3o sugerida pelo prof.)\n"
                 "c\u00e9lulas \u2014 ainda n\u00e3o testadas", y=1.02)
    plt.tight_layout()
    plt.savefig("resultados/heatmap_expandido.png", dpi=140)
    plt.close()


def plot_boxplots_expandido(data_original, data_expandido, data_v3=None, data_v4=None):
    """Boxplots lado a lado mostrando as configs originais, expandidas, v3 e v4."""
    originais = sorted(data_original["configs"], key=lambda c: (c["N"], c["K"]))
    expandidas = list(data_expandido["configs"])
    v3 = list(data_v3["configs"]) if data_v3 is not None else []
    v4 = list(data_v4["configs"]) if data_v4 is not None else []
    # Remove duplicatas
    chaves_orig = {(c["N"], c["K"]) for c in originais}
    novas_exp = [c for c in expandidas if (c["N"], c["K"]) not in chaves_orig]
    chaves_acum = chaves_orig | {(c["N"], c["K"]) for c in novas_exp}
    novas_v3 = [c for c in v3 if (c["N"], c["K"]) not in chaves_acum]
    chaves_acum = chaves_acum | {(c["N"], c["K"]) for c in novas_v3}
    novas_v4 = [c for c in v4 if (c["N"], c["K"]) not in chaves_acum]

    todas = originais + novas_exp + novas_v3 + novas_v4
    dados = [c["custos_finais"] for c in todas]
    labels = [f"N={c['N']}\nK={c['K']}" for c in todas]
    cores = (["#cccccc"] * len(originais)
             + ["#aec7e8"] * len(novas_exp)
             + ["#9467bd"] * len(novas_v3)
             + ["#ff9896"] * len(novas_v4))

    melhor_idx = min(range(len(todas)), key=lambda i: todas[i]["media"])

    fig, ax = plt.subplots(figsize=(15, 5))
    bp = ax.boxplot(dados, tick_labels=labels, widths=0.6, patch_artist=True)
    for i, patch in enumerate(bp["boxes"]):
        patch.set_facecolor(cores[i]); patch.set_alpha(0.7)
    bp["boxes"][melhor_idx].set_facecolor("#2ca02c"); bp["boxes"][melhor_idx].set_alpha(0.9)

    # Linhas verticais separando original / expansao / v3 / v4
    sep = len(originais)
    ax.axvline(sep + 0.5, color="gray", linestyle=":", lw=1.0)
    sep += len(novas_exp)
    if novas_v3 or novas_v4:
        ax.axvline(sep + 0.5, color="gray", linestyle=":", lw=1.0)
    sep += len(novas_v3)
    if novas_v4:
        ax.axvline(sep + 0.5, color="gray", linestyle=":", lw=1.0)
    ax.set_ylabel("Custo final")
    legenda_extra = ""
    if novas_v3:
        legenda_extra += "  |  roxo = v3"
    if novas_v4:
        legenda_extra += "  |  rosa = v4"
    ax.set_title("Sintonia expandida: distribui\u00e7\u00f5es por configura\u00e7\u00e3o\n"
                 f"cinza = original  |  azul = expans\u00e3o{legenda_extra}  |  verde = melhor m\u00e9dia global")
    ax.grid(alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig("resultados/boxplots_expandido.png", dpi=140)
    plt.close()


def plot_curva_marginal(data_original, data_expandido, data_v3=None, data_v4=None):
    """
    Curva mostrando o ganho marginal de aumentar N com K=5 fixo.
    Util para visualizar o joelho da curva.
    """
    todos = list(data_original["configs"]) + list(data_expandido["configs"])
    if data_v3 is not None:
        todos = todos + list(data_v3["configs"])
    if data_v4 is not None:
        todos = todos + list(data_v4["configs"])
    # Filtra so K=5
    so_k5 = [c for c in todos if c["K"] == 5]
    # Remove duplicatas
    visto = set()
    unicos = []
    for c in so_k5:
        if c["N"] not in visto:
            visto.add(c["N"])
            unicos.append(c)
    unicos = sorted(unicos, key=lambda c: c["N"])

    Ns = [c["N"] for c in unicos]
    medias = [c["media"] for c in unicos]
    desvios = [c["desvio"] for c in unicos]
    minimos = [c["minimo"] for c in unicos]

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.errorbar(Ns, medias, yerr=desvios, fmt="o-", color="#1f77b4",
                capsize=6, lw=1.5, label="m\u00e9dia \u00b1 desvio",
                markersize=8)
    ax.plot(Ns, minimos, "v--", color="red", lw=1.0, alpha=0.7,
            label="m\u00ednimo observado")

    # Anota ganhos marginais
    for i in range(1, len(Ns)):
        ganho = (1 - medias[i] / medias[i-1]) * 100
        x_mid = (Ns[i] + Ns[i-1]) / 2
        y_mid = (medias[i] + medias[i-1]) / 2
        ax.annotate(f"{ganho:+.1f}%", xy=(x_mid, y_mid),
                    xytext=(x_mid, y_mid + 400),
                    ha="center", fontsize=8,
                    color="gray")

    ax.set_xlabel("N (popula\u00e7\u00e3o)")
    ax.set_ylabel("Custo final")
    ax.set_xscale("log")
    ax.set_title("Ganho marginal vs popula\u00e7\u00e3o (K=5 fixo)\n"
                 "porcentagens mostram redu\u00e7\u00e3o entre patamares consecutivos")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3, which="both")
    for N, m in zip(Ns, medias):
        ax.text(N, m - 250, f"{m:.0f}", ha="center", fontsize=9,
                fontweight="bold")
    plt.tight_layout()
    plt.savefig("resultados/curva_marginal.png", dpi=140)
    plt.close()


def plot_melhor_rota_expandida(data_expandido, csv_path, data_v3=None, data_v4=None):
    """Melhor rota 3D entre todas as configs expandidas (incluindo v3 e v4)."""
    cidades, grupos, _, _ = preparar_problema(csv_path)
    candidatos = list(data_expandido["configs"])
    if data_v3 is not None:
        candidatos = candidatos + list(data_v3["configs"])
    if data_v4 is not None:
        candidatos = candidatos + list(data_v4["configs"])
    melhor_cfg = min(candidatos, key=lambda c: c["minimo"])
    rota_idx = np.array(melhor_cfg["melhor_rota"])
    rota = np.concatenate([[0], rota_idx, [0]])
    coord = cidades[rota]
    custo = float(melhor_cfg["minimo"])

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    cores = {1: "#1f77b4", 2: "#2ca02c", 3: "#9467bd", 4: "#d62728"}
    for g in [1, 2, 3, 4]:
        mask = grupos == g
        ax.scatter(cidades[mask, 0], cidades[mask, 1], cidades[mask, 2],
                   c=cores[g], s=18, alpha=0.7, label=f"Grupo {g}")
    ax.scatter([cidades[0, 0]], [cidades[0, 1]], [cidades[0, 2]],
               c="black", s=180, marker="*", edgecolor="yellow",
               linewidth=1.2, zorder=10, label="Origem")
    ax.plot(coord[:, 0], coord[:, 1], coord[:, 2], "k-", lw=0.6, alpha=0.5)
    ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
    ax.set_title(f"Melhor rota da sintonia expandida "
                 f"(N={melhor_cfg['N']}, K={melhor_cfg['K']})\ncusto = {custo:.0f}")
    ax.legend(fontsize=8, loc="upper left")
    plt.tight_layout()
    plt.savefig("resultados/expandida_melhor_rota_3d.png", dpi=140)
    plt.close()


if __name__ == "__main__":
    os.makedirs("resultados", exist_ok=True)

    # --- Etapa 1: Sintonia ---
    data = carregar()
    print("[sintonia] Heatmap N x K...")
    plot_heatmap(data)
    print("[sintonia] Boxplots...")
    plot_boxplots(data)
    print("[sintonia] Grid de converg\u00eancia 3x3...")
    plot_convergencia_grid(data)
    print("[sintonia] Melhor rota 3D...")
    plot_melhor_rota(data, "CaixeiroGruposGA.csv")

    # --- Etapa 2: Elitismo ---
    if os.path.exists("resultados/elitismo_completa.json"):
        data_el = carregar_elitismo()
        print("[elitismo] Boxplot por N_elite...")
        plot_elitismo_boxplot(data_el)
        print("[elitismo] Curvas de converg\u00eancia comparativas...")
        plot_elitismo_convergencia(data_el)
        print("[elitismo] Barras m\u00e9dia \u00b1 desvio...")
        plot_elitismo_barras(data_el)
        print("[elitismo] Moda de gera\u00e7\u00f5es para solu\u00e7\u00e3o aceit\u00e1vel...")
        info = plot_elitismo_moda_geracoes(data_el)
        print(f"  limiar = {info['limiar']:.0f}  (melhor global "
              f"{info['melhor_global']:.0f} + {info['tolerancia']*100:.0f}%)")
        print(f"  modas por N_elite: {info['modas']}")
        print("[elitismo] Melhor rota 3D...")
        plot_elitismo_melhor_rota(data_el, "CaixeiroGruposGA.csv")
    else:
        print("\n(skip: resultados/elitismo_completa.json n\u00e3o encontrado)")

    # --- Etapa 3: Analise final (5000 geracoes) ---
    if os.path.exists("resultados/final_5000ger.json"):
        data_final = carregar_final()
        data_el_ref = carregar_elitismo() if os.path.exists(
            "resultados/elitismo_completa.json") else None
        print("[final] Curva de converg\u00eancia (com compara\u00e7\u00e3o)...")
        plot_final_convergencia(data_final, data_el_ref)
        print("[final] Boxplot comparativo 1000 vs 5000 ger...")
        plot_final_boxplot_comparativo(data_final, data_el_ref)
        print("[final] Melhor rota 3D...")
        plot_final_melhor_rota(data_final, "CaixeiroGruposGA.csv")

    # --- Etapa 4: Sintonia expandida (extremos sugeridos pelo professor) ---
    if os.path.exists("resultados/sintonia_expandida.json"):
        data_exp = carregar_expandida()
        data_v3 = carregar_v3()
        data_v4 = carregar_v4()
        print("[expandida] Heatmap combinado original + expandido...")
        plot_heatmap_expandido(data, data_exp, data_v3, data_v4)
        print("[expandida] Boxplots combinados...")
        plot_boxplots_expandido(data, data_exp, data_v3, data_v4)
        print("[expandida] Curva marginal (ganho vs N com K=5)...")
        plot_curva_marginal(data, data_exp, data_v3, data_v4)
        print("[expandida] Melhor rota 3D da expansao...")
        plot_melhor_rota_expandida(data_exp, "CaixeiroGruposGA.csv", data_v3, data_v4)

    print("\nFiguras salvas em resultados/")
