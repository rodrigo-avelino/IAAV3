"""
Figuras da Tarefa 2 — 8 Rainhas com Têmpera Simulada.

1. Heatmap T0 x alpha — taxa de sucesso na sintonia.
2. Heatmap T0 x alpha — custo (mediana de iterações por sucesso).
3. Evolução de uma execução típica: aptidão e temperatura ao longo do tempo.
4. Histograma de iterações por execução bem-sucedida na busca das 92 soluções.
5. Curva de descoberta: número de soluções únicas vs número de execuções.
6. Histograma dos gaps entre descobertas de novas soluções.
"""
import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from tempera import tempera_simulada


def plot_heatmaps_sintonia():
    data = np.load("resultados/sintonia_grid.npz")
    T0 = data["T0_lista"]
    alphas = data["alpha_lista"]
    sucesso = data["mapa_sucesso"]
    mediana = data["mapa_iter_mediana"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    im1 = ax1.imshow(sucesso, cmap="RdYlGn", vmin=0, vmax=100, aspect="auto")
    ax1.set_xticks(range(len(alphas))); ax1.set_xticklabels([f"{a:.2f}" for a in alphas])
    ax1.set_yticks(range(len(T0))); ax1.set_yticklabels([f"{t:.0f}" for t in T0])
    ax1.set_xlabel(r"$\alpha$"); ax1.set_ylabel(r"$T_0$")
    ax1.set_title("Taxa de sucesso (%)")
    for i in range(len(T0)):
        for j in range(len(alphas)):
            v = sucesso[i, j]
            cor = "white" if v < 30 or v > 70 else "black"
            ax1.text(j, i, f"{v:.0f}", ha="center", va="center",
                     color=cor, fontweight="bold", fontsize=9)
    fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04, label="%")

    im2 = ax2.imshow(mediana, cmap="YlOrRd", aspect="auto")
    ax2.set_xticks(range(len(alphas))); ax2.set_xticklabels([f"{a:.2f}" for a in alphas])
    ax2.set_yticks(range(len(T0))); ax2.set_yticklabels([f"{t:.0f}" for t in T0])
    ax2.set_xlabel(r"$\alpha$"); ax2.set_ylabel(r"$T_0$")
    ax2.set_title("Mediana de iterações por sucesso")
    vmax = mediana.max()
    for i in range(len(T0)):
        for j in range(len(alphas)):
            v = mediana[i, j]
            cor = "white" if v > 0.6 * vmax else "black"
            ax2.text(j, i, f"{v:.0f}", ha="center", va="center",
                     color=cor, fontweight="bold", fontsize=9)
    fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04, label="iter")

    plt.tight_layout()
    plt.savefig("resultados/heatmap_sintonia.png", dpi=140)
    plt.close()


def plot_evolucao_tipica(seed=42):
    rng = np.random.default_rng(seed)
    r = tempera_simulada(T0=100.0, alpha=0.9, max_it=5000, rng=rng)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 4.5), sharex=True)

    ax1.plot(r["hist_f"], color="#1f77b4", lw=1.0)
    ax1.axhline(28, color="k", linestyle=":", lw=1.0, label="ótimo (f=28)")
    ax1.set_ylabel(r"Aptidão $f(\mathbf{x})$")
    ax1.set_title(f"Evolução de uma execução típica – T0=100, α=0.9 "
                  f"({'sucesso' if r['sucesso'] else 'falhou'} em {r['iteracoes']} iter)")
    ax1.legend(fontsize=8); ax1.grid(alpha=0.3)

    ax2.plot(r["hist_T"], color="#d62728", lw=1.0)
    ax2.set_yscale("log")
    ax2.set_xlabel("Iteração"); ax2.set_ylabel(r"$T$ (log)")
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig("resultados/evolucao_tipica.png", dpi=140)
    plt.close()


def plot_busca_92():
    with open("resultados/busca_92.json") as fh:
        d = json.load(fh)

    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))

    # (a) Curva cumulativa de descobertas
    gaps = d["gaps_descobertas"]
    execucoes_descoberta = np.cumsum(gaps)
    axes[0].plot(execucoes_descoberta, range(1, len(gaps) + 1),
                 color="#1f77b4", lw=1.2)
    axes[0].set_xlabel("Execução")
    axes[0].set_ylabel("Soluções únicas acumuladas")
    axes[0].set_title("Curva de descoberta das 92 soluções")
    axes[0].grid(alpha=0.3)
    axes[0].axhline(92, color="k", linestyle=":", lw=1.0)

    # (b) Histograma de gaps
    axes[1].hist(gaps, bins=30, color="#2ca02c", edgecolor="black")
    axes[1].set_xlabel("Gap (execuções entre novas soluções)")
    axes[1].set_ylabel("Frequência")
    axes[1].set_title(f"Gaps entre descobertas (média = {np.mean(gaps):.1f})")
    axes[1].grid(alpha=0.3)

    # (c) Histograma de iterações por execução bem-sucedida
    iters = d["iter_por_execucao_bem_sucedida"]
    axes[2].hist(iters, bins=40, color="#ff7f0e", edgecolor="black")
    axes[2].set_xlabel("Iterações até atingir f=28")
    axes[2].set_ylabel("Frequência")
    axes[2].set_title(f"Custo por execução bem-sucedida\n(mediana = {np.median(iters):.0f}, "
                      f"média = {np.mean(iters):.0f})")
    axes[2].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig("resultados/busca_92_analise.png", dpi=140)
    plt.close()


if __name__ == "__main__":
    os.makedirs("resultados", exist_ok=True)
    print("  Heatmaps de sintonia...")
    plot_heatmaps_sintonia()
    print("  Evolução típica...")
    plot_evolucao_tipica()
    print("  Análise da busca pelas 92 soluções...")
    plot_busca_92()
    print("Figuras salvas em resultados/.")
