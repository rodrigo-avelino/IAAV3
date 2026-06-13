"""
Busca Local Aleatoria (LRS) para a Tarefa 2.3.

Implementacao identica a da Tarefa 1, generalizada para n variaveis e
adaptada para fornecer historico de avaliacoes (para comparacao justa
com o GA por orcamento de avaliacoes).

Memoria utilizada:
    O(n) para x_best + 1 candidato por iteracao.
    Em contraste, GA usa O(N * n) (populacao inteira) + buffers
    temporarios para filhos. Para N=100 e n=50, o GA usa 100x mais
    memoria que o LRS.

Parametros:
    sigma:    desvio-padrao da perturbacao gaussiana.
    max_it:   numero maximo de iteracoes (= numero de avaliacoes de f).
    paciencia: se nao melhora apos esse numero de iteracoes, para.
"""
import numpy as np


def lrs(f, n_var, inf, sup, sigma=0.5, max_it=100000, paciencia=None, rng=None):
    """
    Retorna dict com:
        melhor_x, melhor_custo, hist_melhor, hist_media (nao se aplica),
        iteracoes, parou_por, avaliacoes_totais.
    """
    if rng is None:
        rng = np.random.default_rng()

    # Ponto inicial uniforme nos bounds
    x_best = rng.uniform(inf, sup)
    f_best = float(f(x_best))

    hist_melhor = [f_best]
    sem_melhora = 0
    parou_por = "max_it"

    for it in range(max_it):
        ruido = rng.normal(0.0, sigma, size=n_var)
        y = np.clip(x_best + ruido, inf, sup)
        fy = float(f(y))

        if fy < f_best:
            x_best, f_best = y, fy
            sem_melhora = 0
        else:
            sem_melhora += 1

        hist_melhor.append(f_best)

        if paciencia is not None and sem_melhora >= paciencia:
            parou_por = "estagnacao"
            break

    return {
        "melhor_x": x_best,
        "melhor_custo": f_best,
        "hist_melhor": hist_melhor,
        "iteracoes": it + 1,
        "parou_por": parou_por,
        "avaliacoes_totais": it + 2,  # 1 do ponto inicial + (it+1) candidatos
    }


def sa(f, n_var, inf, sup, sigma=0.5, T0=100.0, alpha=0.99,
       max_it=100000, rng=None):
    """
    Tempera Simulada adaptada para domino continuo.

    Estado: vetor de reais x in [inf, sup]^n.
    Vizinhanca: x + N(0, sigma^2 I).
    Criterio de Metropolis para aceitacao.
    Resfriamento geometrico: T_{k+1} = alpha * T_k.

    Parametros:
        sigma: desvio-padrao da perturbacao gaussiana.
        T0:    temperatura inicial.
        alpha: taxa de resfriamento (0 < alpha < 1).
        max_it: maximo de iteracoes (= numero de avaliacoes).

    Retorna:
        dict com melhor_x, melhor_custo, hist_melhor, hist_T, iteracoes,
        avaliacoes_totais, taxa_aceitacao.
    """
    if rng is None:
        rng = np.random.default_rng()

    x_atual = rng.uniform(inf, sup)
    f_atual = float(f(x_atual))
    x_best = x_atual.copy()
    f_best = f_atual

    T = T0
    hist_melhor = [f_best]
    hist_T = [T]
    aceitos = 0

    for it in range(max_it):
        ruido = rng.normal(0.0, sigma, size=n_var)
        y = np.clip(x_atual + ruido, inf, sup)
        fy = float(f(y))
        delta = fy - f_atual

        if delta <= 0:
            aceito = True
        else:
            aceito = (rng.random() < np.exp(-delta / max(T, 1e-300)))

        if aceito:
            x_atual, f_atual = y, fy
            aceitos += 1
            if fy < f_best:
                x_best, f_best = y, fy

        T = alpha * T
        hist_melhor.append(f_best)
        hist_T.append(T)

    return {
        "melhor_x": x_best,
        "melhor_custo": f_best,
        "hist_melhor": hist_melhor,
        "hist_T": hist_T,
        "iteracoes": max_it,
        "avaliacoes_totais": max_it + 1,
        "taxa_aceitacao": aceitos / max_it,
    }


if __name__ == "__main__":
    from funcao import rastrigin, INF, SUP, N_VARIAVEIS
    import time

    rng = np.random.default_rng(42)
    t0 = time.time()
    r = lrs(rastrigin, N_VARIAVEIS, INF, SUP,
            sigma=0.5, max_it=100000, rng=rng)
    t = time.time() - t0
    print(f"LRS sigma=0.5, max_it=100k: {t:.1f}s")
    print(f"  custo final:        {r['melhor_custo']:.2f}")
    print(f"  iteracoes:          {r['iteracoes']}")
    print(f"  avaliacoes totais:  {r['avaliacoes_totais']}")
    print(f"  parou por:          {r['parou_por']}")
