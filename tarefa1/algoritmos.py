"""
Algoritmos de busca para problemas contínuos com restrição de caixa.

Cada algoritmo retorna um dicionário com:
- x_best: melhor solução encontrada
- f_best: valor da função objetivo em x_best
- hist_f: histórico de f_best ao longo das iterações (sempre comprimento max_it+1)
- hist_x: histórico de x_best ao longo das iterações
- iteracoes_efetivas: quantas iterações rodaram de fato (antes da parada antecipada)
- parou_por: "max_it" ou "patience"

Convenções do enunciado seguidas:
- HC: ponto inicial é o limite inferior de cada variável. Vizinhança é a caixa
  |x_best - y| <= epsilon. O candidato é amostrado uniformemente nessa caixa.
- LRS: ponto inicial é amostrado uniformemente no domínio. Candidato é
  x_best + n, com n ~ N(0, sigma * I). 0 < sigma < 1.
- GRS: cada candidato é amostrado uniformemente no domínio (sem usar x_best).
- Todos verificam restrição de caixa via clipping nos limites.
- Parada antecipada: t iterações consecutivas sem melhoria de f_best.
"""
import numpy as np


def _melhor(f_novo, f_velho, modo):
    """Retorna True se f_novo é estritamente melhor que f_velho dado o modo."""
    return f_novo < f_velho if modo == "min" else f_novo > f_velho


def _registrar(hist_f, hist_x, f_best, x_best, max_it):
    """Preenche histórico até max_it+1 com o último valor (para padronizar comprimento)."""
    while len(hist_f) < max_it + 1:
        hist_f.append(f_best)
        hist_x.append(x_best.copy())


def hill_climbing(f, bounds, epsilon, modo="min", max_it=1000, t_patience=100, rng=None):
    """
    Subida/Descida de Encosta (Hill Climbing).

    Vizinhança: caixa de raio epsilon em torno de x_best (norma infinito).
    Ponto inicial: limite inferior do domínio.
    """
    if rng is None:
        rng = np.random.default_rng()
    bounds = np.asarray(bounds, dtype=float)
    inf, sup = bounds[:, 0], bounds[:, 1]

    x_best = inf.copy()
    f_best = f(x_best)

    hist_f = [f_best]
    hist_x = [x_best.copy()]
    sem_melhoria = 0
    parou_por = "max_it"

    for it in range(max_it):
        # Candidato uniformemente na caixa de raio epsilon, depois clipping
        baixo = np.maximum(x_best - epsilon, inf)
        alto = np.minimum(x_best + epsilon, sup)
        y = rng.uniform(baixo, alto)
        fy = f(y)

        if _melhor(fy, f_best, modo):
            x_best, f_best = y, fy
            sem_melhoria = 0
        else:
            sem_melhoria += 1

        hist_f.append(f_best)
        hist_x.append(x_best.copy())

        if sem_melhoria >= t_patience:
            parou_por = "patience"
            break

    _registrar(hist_f, hist_x, f_best, x_best, max_it)
    return {
        "x_best": x_best,
        "f_best": f_best,
        "hist_f": hist_f,
        "hist_x": hist_x,
        "iteracoes_efetivas": it + 1,
        "parou_por": parou_por,
    }


def local_random_search(f, bounds, sigma, modo="min", max_it=1000, t_patience=100, rng=None):
    """
    Busca Local Aleatória (LRS).

    Candidato: y = x_best + n, n ~ N(0, sigma * I).
    Ponto inicial: amostrado uniformemente no domínio.
    """
    if rng is None:
        rng = np.random.default_rng()
    bounds = np.asarray(bounds, dtype=float)
    inf, sup = bounds[:, 0], bounds[:, 1]

    x_best = rng.uniform(inf, sup)
    f_best = f(x_best)

    hist_f = [f_best]
    hist_x = [x_best.copy()]
    sem_melhoria = 0
    parou_por = "max_it"

    for it in range(max_it):
        n = rng.normal(0.0, sigma, size=len(bounds))
        y = np.clip(x_best + n, inf, sup)
        fy = f(y)

        if _melhor(fy, f_best, modo):
            x_best, f_best = y, fy
            sem_melhoria = 0
        else:
            sem_melhoria += 1

        hist_f.append(f_best)
        hist_x.append(x_best.copy())

        if sem_melhoria >= t_patience:
            parou_por = "patience"
            break

    _registrar(hist_f, hist_x, f_best, x_best, max_it)
    return {
        "x_best": x_best,
        "f_best": f_best,
        "hist_f": hist_f,
        "hist_x": hist_x,
        "iteracoes_efetivas": it + 1,
        "parou_por": parou_por,
    }


def global_random_search(f, bounds, modo="min", max_it=1000, t_patience=None, rng=None):
    """
    Busca Aleatória Global (GRS).

    Cada candidato é amostrado uniformemente em todo o domínio, sem usar x_best.
    Parada antecipada desativada por padrão: como cada amostra é independente,
    a probabilidade de melhorar continua positiva ao longo de todas as iterações,
    e usar patience baixa neste caso descarta a contribuição esperada da cauda.
    """
    if rng is None:
        rng = np.random.default_rng()
    bounds = np.asarray(bounds, dtype=float)
    inf, sup = bounds[:, 0], bounds[:, 1]

    x_best = rng.uniform(inf, sup)
    f_best = f(x_best)

    hist_f = [f_best]
    hist_x = [x_best.copy()]
    sem_melhoria = 0
    parou_por = "max_it"

    for it in range(max_it):
        y = rng.uniform(inf, sup)
        fy = f(y)

        if _melhor(fy, f_best, modo):
            x_best, f_best = y, fy
            sem_melhoria = 0
        else:
            sem_melhoria += 1

        hist_f.append(f_best)
        hist_x.append(x_best.copy())

        if t_patience is not None and sem_melhoria >= t_patience:
            parou_por = "patience"
            break

    _registrar(hist_f, hist_x, f_best, x_best, max_it)
    return {
        "x_best": x_best,
        "f_best": f_best,
        "hist_f": hist_f,
        "hist_x": hist_x,
        "iteracoes_efetivas": it + 1,
        "parou_por": parou_por,
    }
