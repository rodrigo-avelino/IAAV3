"""
Figuras da Tarefa 2.3 - comparacao GA real-coded vs LRS no Rastrigin n=50.

1. Barras: sintonia LRS (media +/- desvio por sigma).
2. Barras: sintonia GA (media +/- desvio por N).
3. Boxplot comparativo de todas as 9 configuracoes.
4. Boxplot do MELHOR de cada (LRS vs GA), comparacao direta.
5. Curvas de convergencia do melhor LRS vs melhor GA (eixo X = avaliacoes).
"""
import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def carregar():
    with open("resultados/lrs_sintonia.json") as fh:
        lrs = json.load(fh)["configs"]
    with open("resultados/ga_sintonia.json") as fh:
        ga = json.load(fh)["configs"]
    return lrs, ga


def carregar_sintonia_fina():
    path = "resultados/sintonia_fina.json"
    if not os.path.exists(path):
        return None
    with open(path) as fh:
        return json.load(fh)


def carregar_expandida():
    path = "resultados/sintonia_expandida.json"
    if not os.path.exists(path):
        return None
    with open(path) as fh:
        return json.load(fh)


def plot_barras_lrs(lrs):
    sigmas = [c["sigma"] for c in lrs]
    medias = [c["media"] for c in lrs]
    desvios = [c["desvio"] for c in lrs]
    minimos = [c["minimo"] for c in lrs]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    x = np.arange(len(sigmas))
    ax.bar(x, medias, yerr=desvios, capsize=6,
           color="#1f77b4", alpha=0.7, edgecolor="black",
           label="m\u00e9dia \u00b1 desvio")
    ax.scatter(x, minimos, color="red", marker="v", s=70, zorder=5,
               label="m\u00ednimo observado")
    melhor_idx = int(np.argmin(medias))
    ax.bar(melhor_idx, medias[melhor_idx], yerr=desvios[melhor_idx],
           capsize=6, color="#2ca02c", alpha=0.9, edgecolor="black")

    ax.set_xticks(x); ax.set_xticklabels([f"{s:.2f}" for s in sigmas])
    ax.set_xlabel(r"$\sigma$ (LRS)")
    ax.set_ylabel("Custo final no Rastrigin (n=50)")
    ax.set_title("Sintonia LRS \u2014 30 rodadas por configura\u00e7\u00e3o\n"
                 "verde = melhor m\u00e9dia")
    ax.legend()
    ax.grid(alpha=0.3, axis="y")
    for i, m in enumerate(medias):
        ax.text(i, m + desvios[i] + 10, f"{m:.0f}",
                ha="center", fontsize=9)
    plt.tight_layout()
    plt.savefig("resultados/lrs_sintonia_barras.png", dpi=140)
    plt.close()


def plot_barras_ga(ga):
    Ns = [c["N"] for c in ga]
    medias = [c["media"] for c in ga]
    desvios = [c["desvio"] for c in ga]
    minimos = [c["minimo"] for c in ga]
    max_gens = [c["max_gen"] for c in ga]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    x = np.arange(len(Ns))
    ax.bar(x, medias, yerr=desvios, capsize=6,
           color="#1f77b4", alpha=0.7, edgecolor="black",
           label="m\u00e9dia \u00b1 desvio")
    ax.scatter(x, minimos, color="red", marker="v", s=70, zorder=5,
               label="m\u00ednimo observado")
    melhor_idx = int(np.argmin(medias))
    ax.bar(melhor_idx, medias[melhor_idx], yerr=desvios[melhor_idx],
           capsize=6, color="#2ca02c", alpha=0.9, edgecolor="black")

    labels = [f"N={n}\nmax_gen={mg}" for n, mg in zip(Ns, max_gens)]
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_xlabel("Configura\u00e7\u00e3o do GA")
    ax.set_ylabel("Custo final no Rastrigin (n=50)")
    ax.set_title("Sintonia GA \u2014 30 rodadas por configura\u00e7\u00e3o "
                 "(or\u00e7amento 100k avalia\u00e7\u00f5es)\nverde = melhor m\u00e9dia")
    ax.legend()
    ax.grid(alpha=0.3, axis="y")
    for i, m in enumerate(medias):
        ax.text(i, m + desvios[i] + 10, f"{m:.0f}",
                ha="center", fontsize=9)
    plt.tight_layout()
    plt.savefig("resultados/ga_sintonia_barras.png", dpi=140)
    plt.close()


def plot_boxplot_todas(lrs, ga):
    """Boxplot lado a lado de todas as 9 configuracoes."""
    dados = [c["custos_finais"] for c in lrs] + \
            [c["custos_finais"] for c in ga]
    labels = ([f"LRS\nσ={c['sigma']:.2f}" for c in lrs] +
              [f"GA\nN={c['N']}" for c in ga])

    fig, ax = plt.subplots(figsize=(11, 4.5))
    bp = ax.boxplot(dados, tick_labels=labels, widths=0.6, patch_artist=True)
    for i, patch in enumerate(bp["boxes"]):
        cor = "#aec7e8" if i < len(lrs) else "#ffbb78"
        patch.set_facecolor(cor); patch.set_alpha(0.8)

    # Destaca melhor de cada
    melhor_lrs_idx = int(np.argmin([c["media"] for c in lrs]))
    melhor_ga_idx = int(np.argmin([c["media"] for c in ga])) + len(lrs)
    bp["boxes"][melhor_lrs_idx].set_facecolor("#1f77b4")
    bp["boxes"][melhor_ga_idx].set_facecolor("#ff7f0e")

    ax.axvline(len(lrs) + 0.5, color="gray", linestyle=":", lw=1.0)
    ax.set_ylabel("Custo final no Rastrigin (n=50)")
    ax.set_title("Distribui\u00e7\u00e3o do custo final por configura\u00e7\u00e3o (30 rodadas)\n"
                 "destacado: melhor de cada m\u00e9todo")
    ax.grid(alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig("resultados/boxplot_todas_configs.png", dpi=140)
    plt.close()


def plot_boxplot_melhor_vs_melhor(lrs, ga):
    melhor_lrs = min(lrs, key=lambda c: c["media"])
    melhor_ga = min(ga, key=lambda c: c["media"])

    fig, ax = plt.subplots(figsize=(6, 4.5))
    bp = ax.boxplot(
        [melhor_lrs["custos_finais"], melhor_ga["custos_finais"]],
        tick_labels=[f"LRS (\u03c3={melhor_lrs['sigma']:.2f})\n"
                     f"m\u00e9d {melhor_lrs['media']:.0f}",
                     f"GA (N={melhor_ga['N']})\n"
                     f"m\u00e9d {melhor_ga['media']:.0f}"],
        widths=0.5, patch_artist=True
    )
    bp["boxes"][0].set_facecolor("#1f77b4"); bp["boxes"][0].set_alpha(0.7)
    bp["boxes"][1].set_facecolor("#ff7f0e"); bp["boxes"][1].set_alpha(0.7)

    ax.set_ylabel("Custo final")
    ax.set_title("Compara\u00e7\u00e3o direta: melhor LRS vs melhor GA\n"
                 "(Rastrigin n=50, 30 rodadas, or\u00e7amento 100k avalia\u00e7\u00f5es)")
    ax.grid(alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig("resultados/melhor_vs_melhor.png", dpi=140)
    plt.close()


def plot_convergencia_comparativa(lrs, ga):
    """Curvas no eixo X = avalia\u00e7\u00f5es (comparacao justa)."""
    melhor_lrs = min(lrs, key=lambda c: c["media"])
    melhor_ga = min(ga, key=lambda c: c["media"])

    # LRS: 1 avaliacao por iteracao => indice da hist = avaliacoes
    H_lrs = np.array(melhor_lrs["hist_melhor_por_rodada"])
    media_lrs = H_lrs.mean(axis=0)
    minimo_lrs = H_lrs.min(axis=0)
    maximo_lrs = H_lrs.max(axis=0)
    avals_lrs = np.arange(len(media_lrs))

    # GA: N avaliacoes por geracao
    H_ga = np.array(melhor_ga["hist_melhor_por_rodada"])
    media_ga = H_ga.mean(axis=0)
    minimo_ga = H_ga.min(axis=0)
    maximo_ga = H_ga.max(axis=0)
    N = melhor_ga["N"]
    avals_ga = np.arange(len(media_ga)) * N

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(avals_lrs, media_lrs, color="#1f77b4", lw=1.5,
            label=f"LRS (\u03c3={melhor_lrs['sigma']:.2f}), m\u00e9d final {melhor_lrs['media']:.0f}")
    ax.fill_between(avals_lrs, minimo_lrs, maximo_lrs,
                    color="#1f77b4", alpha=0.15)

    ax.plot(avals_ga, media_ga, color="#ff7f0e", lw=1.5,
            label=f"GA (N={melhor_ga['N']}), m\u00e9d final {melhor_ga['media']:.0f}")
    ax.fill_between(avals_ga, minimo_ga, maximo_ga,
                    color="#ff7f0e", alpha=0.15)

    ax.set_xlabel("Avalia\u00e7\u00f5es da fun\u00e7\u00e3o objetivo")
    ax.set_ylabel("Custo da melhor solu\u00e7\u00e3o")
    ax.set_title("Converg\u00eancia comparada por or\u00e7amento de avalia\u00e7\u00f5es\n"
                 "linha = m\u00e9dia das 30 rodadas, faixa = envelope min\u2013max")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    ax.set_xscale("log")
    plt.tight_layout()
    plt.savefig("resultados/convergencia_comparativa.png", dpi=140)
    plt.close()


def plot_sintonia_fina_heatmap(data):
    """Heatmap N x sigma_mut com m\u00e9dia e desvio."""
    configs = data["configs"]
    Ns = sorted(set(c["N"] for c in configs))
    sigmas = sorted(set(c["sigma_mut"] for c in configs))
    media = np.zeros((len(Ns), len(sigmas)))
    desvio = np.zeros((len(Ns), len(sigmas)))
    for c in configs:
        i, j = Ns.index(c["N"]), sigmas.index(c["sigma_mut"])
        media[i, j] = c["media"]
        desvio[i, j] = c["desvio"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    im1 = ax1.imshow(media, cmap="RdYlGn_r", aspect="auto")
    ax1.set_xticks(range(len(sigmas))); ax1.set_xticklabels([f"{s:.2f}" for s in sigmas])
    ax1.set_yticks(range(len(Ns))); ax1.set_yticklabels(Ns)
    ax1.set_xlabel(r"$\sigma_{\mathrm{mut}}$"); ax1.set_ylabel("N (popula\u00e7\u00e3o)")
    ax1.set_title("Sintonia fina: custo m\u00e9dio (30 rodadas)")
    for i in range(len(Ns)):
        for j in range(len(sigmas)):
            v = media[i, j]
            cor = "white" if v > media.mean() else "black"
            ax1.text(j, i, f"{v:.0f}", ha="center", va="center",
                     color=cor, fontweight="bold", fontsize=10)
    fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)

    im2 = ax2.imshow(desvio, cmap="YlOrRd", aspect="auto")
    ax2.set_xticks(range(len(sigmas))); ax2.set_xticklabels([f"{s:.2f}" for s in sigmas])
    ax2.set_yticks(range(len(Ns))); ax2.set_yticklabels(Ns)
    ax2.set_xlabel(r"$\sigma_{\mathrm{mut}}$"); ax2.set_ylabel("N (popula\u00e7\u00e3o)")
    ax2.set_title("Desvio-padr\u00e3o")
    for i in range(len(Ns)):
        for j in range(len(sigmas)):
            v = desvio[i, j]
            cor = "white" if v > 0.6 * desvio.max() else "black"
            ax2.text(j, i, f"{v:.0f}", ha="center", va="center",
                     color=cor, fontweight="bold", fontsize=10)
    fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.savefig("resultados/sintonia_fina_heatmap.png", dpi=140)
    plt.close()


def plot_sintonia_fina_comparacao_lrs(data_fina, lrs):
    """Boxplot comparativo: melhor LRS vs melhor GA inicial vs melhor GA sintonia fina."""
    melhor_lrs = min(lrs, key=lambda c: c["media"])
    melhor_fina = data_fina["melhor"]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    bp = ax.boxplot(
        [melhor_lrs["custos_finais"], melhor_fina["custos_finais"]],
        tick_labels=[f"LRS (\u03c3={melhor_lrs['sigma']:.2f})\n"
                     f"m\u00e9d {melhor_lrs['media']:.0f}",
                     f"GA (N={melhor_fina['N']}, "
                     f"\u03c3_mut={melhor_fina['sigma_mut']:.2f})\n"
                     f"m\u00e9d {melhor_fina['media']:.0f}"],
        widths=0.5, patch_artist=True
    )
    bp["boxes"][0].set_facecolor("#1f77b4"); bp["boxes"][0].set_alpha(0.7)
    bp["boxes"][1].set_facecolor("#2ca02c"); bp["boxes"][1].set_alpha(0.7)
    ax.set_ylabel("Custo final")
    ax.set_title("Compara\u00e7\u00e3o final: melhor LRS vs melhor GA (ap\u00f3s sintonia fina)\n"
                 "(Rastrigin n=50, 30 rodadas, 100k avalia\u00e7\u00f5es)")
    ax.grid(alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig("resultados/comparacao_final.png", dpi=140)
    plt.close()


def plot_sintonia_fina_convergencia(data_fina, lrs):
    """Curva de converg\u00eancia do melhor GA novo vs melhor LRS (eixo X = avalia\u00e7\u00f5es)."""
    melhor_lrs = min(lrs, key=lambda c: c["media"])
    melhor_fina = data_fina["melhor"]

    H_lrs = np.array(melhor_lrs["hist_melhor_por_rodada"])
    media_lrs = H_lrs.mean(axis=0)
    minimo_lrs = H_lrs.min(axis=0)
    maximo_lrs = H_lrs.max(axis=0)
    avals_lrs = np.arange(len(media_lrs))

    H_ga = np.array(melhor_fina["hist_melhor_por_rodada"])
    media_ga = H_ga.mean(axis=0)
    minimo_ga = H_ga.min(axis=0)
    maximo_ga = H_ga.max(axis=0)
    N = melhor_fina["N"]
    avals_ga = np.arange(len(media_ga)) * N

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(avals_lrs, media_lrs, color="#1f77b4", lw=1.5,
            label=f"LRS (\u03c3={melhor_lrs['sigma']:.2f}), m\u00e9d {melhor_lrs['media']:.0f}")
    ax.fill_between(avals_lrs, minimo_lrs, maximo_lrs,
                    color="#1f77b4", alpha=0.15)
    ax.plot(avals_ga, media_ga, color="#2ca02c", lw=1.5,
            label=f"GA (N={melhor_fina['N']}, \u03c3_mut={melhor_fina['sigma_mut']:.2f}), "
                  f"m\u00e9d {melhor_fina['media']:.0f}")
    ax.fill_between(avals_ga, minimo_ga, maximo_ga,
                    color="#2ca02c", alpha=0.15)
    ax.set_xlabel("Avalia\u00e7\u00f5es da fun\u00e7\u00e3o objetivo")
    ax.set_ylabel("Custo da melhor solu\u00e7\u00e3o")
    ax.set_title("Converg\u00eancia comparada ap\u00f3s sintonia fina\n"
                 "linha = m\u00e9dia das 30 rodadas, faixa = envelope min\u2013max")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    ax.set_xscale("log")
    plt.tight_layout()
    plt.savefig("resultados/convergencia_final.png", dpi=140)
    plt.close()


# =========================================================================
# Sintonia EXPANDIDA (extremos sugeridos pelo professor)
# =========================================================================

def plot_heatmap_combinado(data_fina, data_expandida):
    """
    Heatmap combinando o grid fina {10,20,30} x {0.10,0.20,0.30}
    com o expandido {5,8,10,15} x {0.30,0.40,0.50,0.70}.
    """
    todos = []
    if data_fina is not None:
        todos.extend(data_fina["configs"])
    todos.extend(data_expandida["configs"])

    visto = set()
    unicos = []
    for c in todos:
        chave = (c["N"], c["sigma_mut"])
        if chave not in visto:
            visto.add(chave)
            unicos.append(c)

    Ns = sorted(set(c["N"] for c in unicos))
    sigmas = sorted(set(c["sigma_mut"] for c in unicos))
    media = np.full((len(Ns), len(sigmas)), np.nan)
    desvio = np.full((len(Ns), len(sigmas)), np.nan)
    for c in unicos:
        i, j = Ns.index(c["N"]), sigmas.index(c["sigma_mut"])
        media[i, j] = c["media"]
        desvio[i, j] = c["desvio"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5))

    im1 = ax1.imshow(media, cmap="RdYlGn_r", aspect="auto")
    ax1.set_xticks(range(len(sigmas))); ax1.set_xticklabels([f"{s:.2f}" for s in sigmas])
    ax1.set_yticks(range(len(Ns))); ax1.set_yticklabels(Ns)
    ax1.set_xlabel(r"$\sigma_{\mathrm{mut}}$"); ax1.set_ylabel("N (popula\u00e7\u00e3o)")
    ax1.set_title("Custo m\u00e9dio \u2014 grid combinado (fina + expandida)")
    for i in range(len(Ns)):
        for j in range(len(sigmas)):
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
    ax2.set_xticks(range(len(sigmas))); ax2.set_xticklabels([f"{s:.2f}" for s in sigmas])
    ax2.set_yticks(range(len(Ns))); ax2.set_yticklabels(Ns)
    ax2.set_xlabel(r"$\sigma_{\mathrm{mut}}$"); ax2.set_ylabel("N (popula\u00e7\u00e3o)")
    ax2.set_title("Desvio-padr\u00e3o \u2014 grid combinado")
    vmax = np.nanmax(desvio)
    for i in range(len(Ns)):
        for j in range(len(sigmas)):
            v = desvio[i, j]
            if np.isnan(v):
                ax2.text(j, i, "\u2014", ha="center", va="center",
                         color="gray", fontsize=10)
            else:
                cor = "white" if v > 0.6 * vmax else "black"
                ax2.text(j, i, f"{v:.0f}", ha="center", va="center",
                         color=cor, fontweight="bold", fontsize=9)
    fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)

    fig.suptitle("Sintonia COMBINADA: c\u00e9lulas \u2014 ainda n\u00e3o testadas", y=1.02)
    plt.tight_layout()
    plt.savefig("resultados/heatmap_combinado.png", dpi=140)
    plt.close()


def plot_expandida_comparacao_lrs(data_expandida, lrs):
    melhor_lrs = min(lrs, key=lambda c: c["media"])
    melhor_exp = data_expandida["melhor"]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    bp = ax.boxplot(
        [melhor_lrs["custos_finais"], melhor_exp["custos_finais"]],
        tick_labels=[f"LRS (\u03c3={melhor_lrs['sigma']:.2f})\n"
                     f"m\u00e9d {melhor_lrs['media']:.0f}",
                     f"GA (N={melhor_exp['N']}, "
                     f"\u03c3_mut={melhor_exp['sigma_mut']:.2f})\n"
                     f"m\u00e9d {melhor_exp['media']:.0f}"],
        widths=0.5, patch_artist=True
    )
    bp["boxes"][0].set_facecolor("#1f77b4"); bp["boxes"][0].set_alpha(0.7)
    bp["boxes"][1].set_facecolor("#2ca02c"); bp["boxes"][1].set_alpha(0.7)
    ax.set_ylabel("Custo final")
    ax.set_title("Compara\u00e7\u00e3o ap\u00f3s sintonia expandida: melhor LRS vs melhor GA\n"
                 "(Rastrigin n=50, 30 rodadas, 100k avalia\u00e7\u00f5es)")
    ax.grid(alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig("resultados/comparacao_expandida.png", dpi=140)
    plt.close()


def plot_expandida_convergencia(data_expandida, lrs):
    melhor_lrs = min(lrs, key=lambda c: c["media"])
    melhor_exp = data_expandida["melhor"]

    H_lrs = np.array(melhor_lrs["hist_melhor_por_rodada"])
    media_lrs = H_lrs.mean(axis=0)
    minimo_lrs = H_lrs.min(axis=0)
    maximo_lrs = H_lrs.max(axis=0)
    avals_lrs = np.arange(len(media_lrs))

    H_ga = np.array(melhor_exp["hist_melhor_por_rodada"])
    media_ga = H_ga.mean(axis=0)
    minimo_ga = H_ga.min(axis=0)
    maximo_ga = H_ga.max(axis=0)
    N = melhor_exp["N"]
    avals_ga = np.arange(len(media_ga)) * N

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(avals_lrs, media_lrs, color="#1f77b4", lw=1.5,
            label=f"LRS (\u03c3={melhor_lrs['sigma']:.2f}), m\u00e9d {melhor_lrs['media']:.0f}")
    ax.fill_between(avals_lrs, minimo_lrs, maximo_lrs,
                    color="#1f77b4", alpha=0.15)
    ax.plot(avals_ga, media_ga, color="#2ca02c", lw=1.5,
            label=f"GA (N={melhor_exp['N']}, \u03c3_mut={melhor_exp['sigma_mut']:.2f}), "
                  f"m\u00e9d {melhor_exp['media']:.0f}")
    ax.fill_between(avals_ga, minimo_ga, maximo_ga,
                    color="#2ca02c", alpha=0.15)
    ax.set_xlabel("Avalia\u00e7\u00f5es da fun\u00e7\u00e3o objetivo")
    ax.set_ylabel("Custo da melhor solu\u00e7\u00e3o")
    ax.set_title("Converg\u00eancia ap\u00f3s sintonia expandida (extremos do professor)\n"
                 "linha = m\u00e9dia das 30 rodadas, faixa = envelope min\u2013max")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    ax.set_xscale("log")
    plt.tight_layout()
    plt.savefig("resultados/convergencia_expandida.png", dpi=140)
    plt.close()


if __name__ == "__main__":
    os.makedirs("resultados", exist_ok=True)
    lrs, ga = carregar()
    print("Barras LRS...");   plot_barras_lrs(lrs)
    print("Barras GA...");    plot_barras_ga(ga)
    print("Boxplot todas configs..."); plot_boxplot_todas(lrs, ga)
    print("Boxplot melhor LRS vs melhor GA..."); plot_boxplot_melhor_vs_melhor(lrs, ga)
    print("Curvas comparativas por avalia\u00e7\u00f5es..."); plot_convergencia_comparativa(lrs, ga)

    fina = carregar_sintonia_fina()
    if fina is not None:
        print("Heatmap sintonia fina N x sigma_mut..."); plot_sintonia_fina_heatmap(fina)
        print("Boxplot LRS vs GA (apos sintonia fina)..."); plot_sintonia_fina_comparacao_lrs(fina, lrs)
        print("Curva converg\u00eancia final..."); plot_sintonia_fina_convergencia(fina, lrs)

    expandida = carregar_expandida()
    if expandida is not None:
        print("Heatmap combinado (fina + expandida)..."); plot_heatmap_combinado(fina, expandida)
        print("Boxplot LRS vs GA (apos expansao)..."); plot_expandida_comparacao_lrs(expandida, lrs)
        print("Curva converg\u00eancia (apos expansao)..."); plot_expandida_convergencia(expandida, lrs)

    print("\nFiguras salvas em resultados/")
