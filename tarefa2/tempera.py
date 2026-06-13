"""
Têmpera Simulada para o problema das 8 Rainhas.

Representação:
    x = vetor de 8 inteiros em [1..8], onde x[c] é a linha da rainha
    na coluna c+1 (uma rainha por coluna, reduzindo o espaço de busca de
    8^8 = 16.777.216 arranjos para enumeração via permutações).

Função de aptidão:
    f(x) = 28 - h(x), onde h(x) é o número de pares de rainhas se atacando
    (mesma linha ou mesma diagonal). f = 28 indica solução válida.
    Este é um problema de MAXIMIZAÇÃO (f -> 28).

Perturbação controlada:
    Sorteia-se uma coluna c uniformemente e troca-se x[c] por uma linha
    diferente da atual. A perturbação não percorre todo o espaço de uma
    só vez: toca em uma única variável e numa linha distinta da atual,
    ou seja, vizinhança de raio 1 na métrica de Hamming restrita.

Decaimento:
    Geométrico: T_{k+1} = alpha * T_k, com 0 < alpha < 1.
    Esta é a forma mais comum apresentada no material de aula.

Critério de aceitação (Metropolis):
    Se delta = f(cand) - f(atual) > 0, aceita.
    Se delta <= 0, aceita com probabilidade exp(delta / T).
    (Note: como é maximização, delta > 0 é melhora.)

Parada:
    f_atual == 28 (ótimo) OU max_it atingido.
"""
import numpy as np


def aptidao(x):
    """Conta pares não-atacantes. Máximo: 28 pares (8C2)."""
    h = 0
    n = len(x)
    for i in range(n):
        for j in range(i + 1, n):
            if x[i] == x[j] or abs(x[i] - x[j]) == abs(i - j):
                h += 1
    return 28 - h


def perturba(x, rng):
    """Move uma rainha para outra linha. Vizinhança controlada."""
    novo = x.copy()
    col = rng.integers(0, 8)
    linha_atual = novo[col]
    # Sorteia uma linha diferente da atual
    nova_linha = rng.integers(1, 9)
    while nova_linha == linha_atual:
        nova_linha = rng.integers(1, 9)
    novo[col] = nova_linha
    return novo


def tempera_simulada(T0=10.0, alpha=0.95, max_it=5000, x_inicial=None, rng=None):
    """
    Retorna dicionário com:
        x_best, f_best, iteracoes, sucesso (bool), hist_f, hist_T.
    """
    if rng is None:
        rng = np.random.default_rng()

    if x_inicial is None:
        x_atual = rng.integers(1, 9, size=8)
    else:
        x_atual = x_inicial.copy()

    f_atual = aptidao(x_atual)
    x_best = x_atual.copy()
    f_best = f_atual

    T = float(T0)
    hist_f = [f_atual]
    hist_T = [T]

    for it in range(max_it):
        if f_atual == 28:
            return {
                "x_best": x_atual,
                "f_best": f_atual,
                "iteracoes": it,
                "sucesso": True,
                "hist_f": hist_f,
                "hist_T": hist_T,
            }

        cand = perturba(x_atual, rng)
        f_cand = aptidao(cand)
        delta = f_cand - f_atual

        if delta > 0:
            x_atual, f_atual = cand, f_cand
        else:
            # delta <= 0: aceita com probabilidade Metropolis
            if rng.random() < np.exp(delta / T):
                x_atual, f_atual = cand, f_cand

        if f_atual > f_best:
            x_best, f_best = x_atual.copy(), f_atual

        hist_f.append(f_atual)
        T = T * alpha
        hist_T.append(T)

    return {
        "x_best": x_best,
        "f_best": f_best,
        "iteracoes": max_it,
        "sucesso": f_best == 28,
        "hist_f": hist_f,
        "hist_T": hist_T,
    }
