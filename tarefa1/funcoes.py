"""
Funções objetivo da Tarefa 1.

Cada função vem acompanhada de:
- bounds: restrição de caixa do domínio
- modo: "min" ou "max"
- otimo: valor ótimo (estimado numericamente em [-100,100]^2 / domínios respectivos)
- ponto_otimo: coordenadas (aproximadas) do ótimo
- tolerancia: distância absoluta de f(x) ao ótimo abaixo da qual consideramos
              que o algoritmo convergiu para o ótimo global. Os valores foram
              calibrados função-a-função observando a topologia (mínimos locais
              próximos do global em Rastrigin e Ackley exigem tolerâncias maiores).
"""
import numpy as np


def f1(x):
    """Esfera. Convexa unimodal."""
    return x[0]**2 + x[1]**2


def f2(x):
    """Soma de duas gaussianas. Maximização. Bimodal."""
    return np.exp(-(x[0]**2 + x[1]**2)) + 2 * np.exp(-((x[0] - 1.7)**2 + (x[1] - 1.7)**2))


def f3(x):
    """Ackley. Minimização. Multimodal com mínimos locais regulares."""
    t1 = -20 * np.exp(-0.2 * np.sqrt(0.5 * (x[0]**2 + x[1]**2)))
    t2 = -np.exp(0.5 * (np.cos(2 * np.pi * x[0]) + np.cos(2 * np.pi * x[1])))
    return t1 + t2 + 20 + np.exp(1)


def f4(x):
    """Rastrigin. Minimização. Multimodal com muitos mínimos locais."""
    return (x[0]**2 - 10 * np.cos(2 * np.pi * x[0]) + 10) + \
           (x[1]**2 - 10 * np.cos(2 * np.pi * x[1]) + 10)


def f5(x):
    """Pico estreito + plateau ondulado. Maximização."""
    return (x[0] * np.cos(x[0])) / 20 + 2 * np.exp(-x[0]**2 - (x[1] - 1)**2) + 0.01 * x[0] * x[1]


def f6(x):
    """Crista oscilatória. Maximização."""
    return x[0] * np.sin(4 * np.pi * x[0]) - x[1] * np.sin(4 * np.pi * x[1] + np.pi) + 1


CONFIG = {
    "f1": {
        "funcao": f1,
        "bounds": [(-100.0, 100.0), (-100.0, 100.0)],
        "modo": "min",
        "otimo": 0.0,
        "ponto_otimo": (0.0, 0.0),
        "tolerancia": 1e-2,
    },
    "f2": {
        "funcao": f2,
        "bounds": [(-2.0, 4.0), (-2.0, 5.0)],
        "modo": "max",
        "otimo": 2.0031167461,
        "ponto_otimo": (1.697331, 1.697331),
        "tolerancia": 1e-2,
    },
    "f3": {
        "funcao": f3,
        "bounds": [(-8.0, 8.0), (-8.0, 8.0)],
        "modo": "min",
        "otimo": 0.0,
        "ponto_otimo": (0.0, 0.0),
        "tolerancia": 1e-1,
    },
    "f4": {
        "funcao": f4,
        "bounds": [(-5.12, 5.12), (-5.12, 5.12)],
        "modo": "min",
        "otimo": 0.0,
        "ponto_otimo": (0.0, 0.0),
        # Rastrigin tem mínimo local em (±1, 0)/(0, ±1) com f ≈ 1.0,
        # então a tolerância tem que ser menor que 1 para distinguir o global.
        "tolerancia": 5e-1,
    },
    "f5": {
        "funcao": f5,
        "bounds": [(-10.0, 10.0), (-10.0, 10.0)],
        "modo": "max",
        "otimo": 2.0004499691,
        "ponto_otimo": (0.0150, 1.0000),
        "tolerancia": 5e-2,
    },
    "f6": {
        "funcao": f6,
        "bounds": [(-1.0, 3.0), (-1.0, 3.0)],
        "modo": "max",
        "otimo": 6.2524103806,
        "ponto_otimo": (2.627409, 2.627409),
        # f6 tem cristas locais com f > 5; tolerância 0.1 separa o global.
        "tolerancia": 1e-1,
    },
}


def convergiu(f_obtido, nome_funcao):
    """Retorna True se f_obtido está dentro da tolerância do ótimo global."""
    cfg = CONFIG[nome_funcao]
    return abs(f_obtido - cfg["otimo"]) <= cfg["tolerancia"]
