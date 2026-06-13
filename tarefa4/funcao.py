"""
Funcao Rastrigin com p = 50 variaveis (Tarefa 2.3).

    f(x) = 10*p + sum_i [x_i^2 - 10 cos(2*pi*x_i)]

Otimo global:
    x* = (0, 0, ..., 0)
    f(x*) = 0

Bounds (mesma faixa da Tarefa 1):
    x_i in [-5.12, 5.12]  para todo i

Multimodalidade:
    Cerca de 11^50 minimos locais regulares no espaco; o global em zero
    e cercado por minimos locais em torno de x_i = +/-1, +/-2, ... com
    valores f(x) ~ 1 a 5 (proximos do global).

Tolerancia para considerar "convergiu ao otimo":
    |f - 0| <= 5.0
    Esta tolerancia e propositalmente larga; o minimo local em x_i = +/-1
    tem f(x) ~ 50 para n=50 (1 por dimensao somado), entao tolerancia
    muito apertada exigiria que o algoritmo evitasse TODAS as 50 dimensoes
    serem 1, o que e impraticavel em 100 mil avaliacoes. Tolerancia 5
    captura solucoes "muito boas" (menos de 5/50 = 10% de erro medio por
    dimensao em termos de cosseno).
"""
import numpy as np


N_VARIAVEIS = 50
BOUNDS = np.array([(-5.12, 5.12)] * N_VARIAVEIS)
INF = BOUNDS[:, 0]
SUP = BOUNDS[:, 1]
OTIMO = 0.0
TOLERANCIA = 5.0


def rastrigin(x):
    """Versao vetorizada. Aceita x de shape (n,) ou (N, n)."""
    x = np.asarray(x)
    if x.ndim == 1:
        return 10 * N_VARIAVEIS + np.sum(x**2 - 10 * np.cos(2 * np.pi * x))
    # populacao (N, n)
    return 10 * N_VARIAVEIS + np.sum(x**2 - 10 * np.cos(2 * np.pi * x), axis=1)


def convergiu(f_obtido):
    return abs(f_obtido - OTIMO) <= TOLERANCIA


if __name__ == "__main__":
    print(f"Rastrigin com p = {N_VARIAVEIS} variaveis")
    print(f"Bounds: [{INF[0]}, {SUP[0]}] por dimensao")
    print(f"Otimo:  f(0) = {rastrigin(np.zeros(N_VARIAVEIS)):.4f} (esperado 0)")
    print(f"Canto:  f(inf) = {rastrigin(INF):.4f}")
    print(f"Aleatorio: f(rand) = {rastrigin(np.random.uniform(INF, SUP)):.4f}")
    print(f"Tolerancia para convergencia: {TOLERANCIA}")
    # Norma do canto ao otimo
    print(f"Distancia canto-otimo: {np.linalg.norm(INF):.2f}")
