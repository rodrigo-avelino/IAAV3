"""
Geração de figuras da Tarefa 1.

Quatro tipos de gráfico por função:
1. Curvas de convergência (média ± desvio + melhor + pior rodada).
2. Boxplot comparativo dos três algoritmos.
3. Mapa de calor da sintonia (taxa de convergência por hiperparâmetro).
4. Trajetória 3D da melhor rodada sobre a superfície da função.

E uma figura geral:
5. Heatmap consolidado de taxa de convergência por (função, algoritmo).
"""
import json
import pickle
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from funcoes import CONFIG


def carregar_dados():
    with open("resultados/relatorio_completo.json") as fh:
        rel = json.load(fh)
    with open("resultados/sintonia_detalhada.json") as fh:
        sint = json.load(fh)
    with open("resultados/historicos.pkl", "rb") as fh:
        hists = pickle.load(fh)
    return rel, sint, hists


def plotar_convergencia(nome, rel, hists, modo):
    """Curvas: média das 100 rodadas + envelope min-max em cada iteração."""
    fig, ax = plt.subplots(figsize=(6, 4))

    cores = {"HC": "#1f77b4", "LRS": "#2ca02c", "GRS": "#d62728"}
    for alg in ("HC", "LRS", "GRS"):
        # Histórias têm comprimento max_it+1 = 1001
        H = np.array([h["hist_f"] for h in hists[nome][alg]])
        media = H.mean(axis=0)
        minimo = H.min(axis=0)
        maximo = H.max(axis=0)
        iters = np.arange(len(media))

        ax.plot(iters, media, color=cores[alg], lw=1.5, label=f"{alg} (média)")
        ax.fill_between(iters, minimo, maximo,
                        color=cores[alg], alpha=0.15)

    ax.axhline(CONFIG[nome]["otimo"], color="k", linestyle=":", lw=1.0, label="ótimo")

    ax.set_xlabel("Iteração")
    ax.set_ylabel(r"$f(\mathbf{x}_{\mathrm{best}})$")
    ax.set_title(f"Convergência – {nome.upper()} ({modo})\n"
                 f"linha = média das 100 rodadas, faixa = envelope min–max")
    ax.legend(fontsize=8, loc="best")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"resultados/conv_{nome}.png", dpi=140)
    plt.close()


def plotar_boxplot(nome, rel):
    fig, ax = plt.subplots(figsize=(4, 4))
    dados = [rel[nome][alg + "_dados"]["f"] for alg in ("HC", "LRS", "GRS")]
    ax.boxplot(dados, tick_labels=["HC", "LRS", "GRS"], widths=0.6)
    ax.axhline(CONFIG[nome]["otimo"], color="k", linestyle=":", lw=1.0, label="ótimo")
    ax.set_ylabel(r"$f(\mathbf{x}_{\mathrm{best}})$ final")
    ax.set_title(f"Distribuição – {nome.upper()}")
    ax.grid(alpha=0.3, axis="y")
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(f"resultados/box_{nome}.png", dpi=140)
    plt.close()


def plotar_sintonia(nome, sint):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 3.5))

    # HC: epsilon
    eps = [c["epsilon"] for c in sint[nome]["HC"]["candidatos"]]
    taxas_hc = [c["taxa_convergencia"] * 100 for c in sint[nome]["HC"]["candidatos"]]
    ax1.plot(eps, taxas_hc, "o-", color="#1f77b4")
    ax1.axhline(50, color="k", linestyle=":", lw=1.0, label="limiar 50%")
    eps_esc = sint[nome]["HC"]["escolhido"]["epsilon"]
    ax1.axvline(eps_esc, color="red", linestyle="--", lw=1.0, label=f"escolhido ε={eps_esc:.3f}")
    ax1.set_xscale("log")
    ax1.set_xlabel(r"$\varepsilon$ (escala log)")
    ax1.set_ylabel("Taxa convergência (%)")
    ax1.set_title(f"HC – {nome.upper()}")
    ax1.set_ylim(-5, 105)
    ax1.grid(alpha=0.3)
    ax1.legend(fontsize=7)

    # LRS: sigma
    sigs = [c["sigma"] for c in sint[nome]["LRS"]["candidatos"]]
    taxas_lrs = [c["taxa_convergencia"] * 100 for c in sint[nome]["LRS"]["candidatos"]]
    ax2.plot(sigs, taxas_lrs, "o-", color="#2ca02c")
    ax2.axhline(50, color="k", linestyle=":", lw=1.0, label="limiar 50%")
    sig_esc = sint[nome]["LRS"]["escolhido"]["sigma"]
    ax2.axvline(sig_esc, color="red", linestyle="--", lw=1.0, label=f"escolhido σ={sig_esc:.2f}")
    ax2.set_xlabel(r"$\sigma$")
    ax2.set_ylabel("Taxa convergência (%)")
    ax2.set_title(f"LRS – {nome.upper()}")
    ax2.set_ylim(-5, 105)
    ax2.grid(alpha=0.3)
    ax2.legend(fontsize=7)

    plt.tight_layout()
    plt.savefig(f"resultados/sintonia_{nome}.png", dpi=140)
    plt.close()


def plotar_trajetoria_3d(nome, rel, hists):
    """Trajetória 3D da melhor rodada para cada algoritmo."""
    cfg = CONFIG[nome]
    f, bounds, modo = cfg["funcao"], cfg["bounds"], cfg["modo"]

    fig = plt.figure(figsize=(7, 5.5))
    ax = fig.add_subplot(111, projection="3d")

    # Superfície
    x1 = np.linspace(bounds[0][0], bounds[0][1], 80)
    x2 = np.linspace(bounds[1][0], bounds[1][1], 80)
    X1, X2 = np.meshgrid(x1, x2)
    Z = f([X1, X2])
    ax.plot_surface(X1, X2, Z, cmap="viridis", alpha=0.3, edgecolor="none")

    cores = {"HC": "#1f77b4", "LRS": "#2ca02c", "GRS": "#d62728"}

    for alg in ("HC", "LRS", "GRS"):
        fs = np.array(rel[nome][alg + "_dados"]["f"])
        if modo == "min":
            idx_melhor = int(np.argmin(fs))
        else:
            idx_melhor = int(np.argmax(fs))

        h = hists[nome][alg][idx_melhor]
        xs = np.array(h["hist_x"])
        fs_traj = np.array(h["hist_f"])

        # Trajetória rarefeita para não poluir
        passo = max(1, len(xs) // 100)
        ax.plot(xs[::passo, 0], xs[::passo, 1], fs_traj[::passo],
                color=cores[alg], lw=1.0, alpha=0.7, label=f"{alg}")
        ax.scatter([xs[-1, 0]], [xs[-1, 1]], [fs_traj[-1]],
                   color=cores[alg], s=80, marker="*",
                   edgecolor="k", linewidth=0.5, zorder=5)

    ax.set_xlabel(r"$x_1$")
    ax.set_ylabel(r"$x_2$")
    ax.set_zlabel(r"$f$")
    ax.set_title(f"Trajetória da melhor rodada – {nome.upper()}")
    ax.legend(fontsize=8, loc="upper left")
    plt.tight_layout()
    plt.savefig(f"resultados/traj_{nome}.png", dpi=140)
    plt.close()


def plotar_heatmap_geral(rel):
    """Heatmap consolidado: linha = função, coluna = algoritmo, valor = taxa de conv."""
    funcoes = list(CONFIG.keys())
    algoritmos = ["HC", "LRS", "GRS"]
    matriz = np.array([[rel[f][a]["taxa_convergencia"] * 100 for a in algoritmos]
                       for f in funcoes])

    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    im = ax.imshow(matriz, cmap="RdYlGn", vmin=0, vmax=100, aspect="auto")
    ax.set_xticks(range(len(algoritmos)))
    ax.set_xticklabels(algoritmos)
    ax.set_yticks(range(len(funcoes)))
    ax.set_yticklabels([f.upper() for f in funcoes])

    for i in range(len(funcoes)):
        for j in range(len(algoritmos)):
            val = matriz[i, j]
            cor = "white" if val < 30 or val > 70 else "black"
            ax.text(j, i, f"{val:.0f}%", ha="center", va="center",
                    color=cor, fontweight="bold", fontsize=10)

    ax.set_title("Taxa de convergência ao ótimo (100 rodadas)")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="%")
    plt.tight_layout()
    plt.savefig("resultados/heatmap_geral.png", dpi=140)
    plt.close()


def main():
    rel, sint, hists = carregar_dados()

    for nome in CONFIG:
        cfg = CONFIG[nome]
        modo = cfg["modo"]
        print(f"  Gerando figuras para {nome.upper()}...")
        plotar_convergencia(nome, rel, hists, modo)
        plotar_boxplot(nome, rel)
        plotar_sintonia(nome, sint)
        plotar_trajetoria_3d(nome, rel, hists)

    print("  Gerando heatmap consolidado...")
    plotar_heatmap_geral(rel)
    print("Figuras salvas em resultados/")


if __name__ == "__main__":
    main()
