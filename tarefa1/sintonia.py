"""
Sintonia de hiperparâmetros para HC (epsilon) e LRS (sigma).

Critério de escolha: para cada hiperparâmetro candidato, executa N_RODADAS
independentes e mede a taxa de convergência ao ótimo global (definida como
|f - f_otimo| <= tolerancia, com tolerâncias por função em funcoes.CONFIG).
O hiperparâmetro escolhido é o que MAXIMIZA a taxa de convergência; em caso
de empate, desempata-se pela melhor média (mais próxima do ótimo).

O status indica se o melhor candidato atingiu o limiar TAXA_MINIMA. Casos
com status="abaixo_do_limiar" marcam combinações função/algoritmo em que
nenhum hiperparâmetro permite ao algoritmo convergir consistentemente — uma
limitação inerente do método para aquela topologia, dada a restrição de
ponto inicial imposta pelo enunciado.

GRS não tem hiperparâmetro a sintonizar (o enunciado menciona sigma para GRS,
mas é equívoco: GRS amostra uniformemente em todo o domínio).
"""
import json
import numpy as np

from funcoes import CONFIG, convergiu
from algoritmos import hill_climbing, local_random_search


TAXA_MINIMA = 0.50  # 50% das rodadas precisam convergir
N_RODADAS_SINTONIA = 50
MAX_IT = 1000
T_PATIENCE = 100


# Epsilon candidatos: percentuais da amplitude do domínio (mesma escala
# para todas as funções). Cobre de muito pequeno (0.1% do range) a metade.
PERCENTUAIS_EPSILON = [0.001, 0.005, 0.01, 0.02, 0.05, 0.10, 0.20, 0.35, 0.50]

# Sigma candidatos: o enunciado restringe a 0 < sigma < 1. Cobrimos toda a faixa.
SIGMAS = [0.05, 0.10, 0.20, 0.35, 0.50, 0.65, 0.80, 0.99]


def sintonizar_hc(nome_funcao, rng):
    cfg = CONFIG[nome_funcao]
    f, bounds, modo = cfg["funcao"], cfg["bounds"], cfg["modo"]
    amplitude = bounds[0][1] - bounds[0][0]

    resultados = []
    for p in PERCENTUAIS_EPSILON:
        eps = p * amplitude
        sucessos = 0
        f_finais = []
        for _ in range(N_RODADAS_SINTONIA):
            r = hill_climbing(f, bounds, eps, modo,
                              max_it=MAX_IT, t_patience=T_PATIENCE, rng=rng)
            f_finais.append(r["f_best"])
            if convergiu(r["f_best"], nome_funcao):
                sucessos += 1
        taxa = sucessos / N_RODADAS_SINTONIA
        resultados.append({
            "percentual": p,
            "epsilon": eps,
            "taxa_convergencia": taxa,
            "media_f": float(np.mean(f_finais)),
            "f_finais": f_finais,
        })

    # Escolha: maior taxa de convergência ao ótimo global; em caso de empate,
    # desempate pela melhor média (mais próxima do ótimo segundo o modo).
    # O status registra se o melhor candidato atingiu o limiar TAXA_MINIMA.
    if modo == "min":
        chave = lambda r: (r["taxa_convergencia"], -r["media_f"])
    else:
        chave = lambda r: (r["taxa_convergencia"], r["media_f"])
    escolhido = max(resultados, key=chave)
    status = "ok" if escolhido["taxa_convergencia"] >= TAXA_MINIMA else "abaixo_do_limiar"

    return {
        "algoritmo": "HC",
        "funcao": nome_funcao,
        "candidatos": resultados,
        "escolhido": escolhido,
        "status": status,
    }


def sintonizar_lrs(nome_funcao, rng):
    cfg = CONFIG[nome_funcao]
    f, bounds, modo = cfg["funcao"], cfg["bounds"], cfg["modo"]

    resultados = []
    for sig in SIGMAS:
        sucessos = 0
        f_finais = []
        for _ in range(N_RODADAS_SINTONIA):
            r = local_random_search(f, bounds, sig, modo,
                                    max_it=MAX_IT, t_patience=T_PATIENCE, rng=rng)
            f_finais.append(r["f_best"])
            if convergiu(r["f_best"], nome_funcao):
                sucessos += 1
        taxa = sucessos / N_RODADAS_SINTONIA
        resultados.append({
            "sigma": sig,
            "taxa_convergencia": taxa,
            "media_f": float(np.mean(f_finais)),
            "f_finais": f_finais,
        })

    if modo == "min":
        chave = lambda r: (r["taxa_convergencia"], -r["media_f"])
    else:
        chave = lambda r: (r["taxa_convergencia"], r["media_f"])
    escolhido = max(resultados, key=chave)
    status = "ok" if escolhido["taxa_convergencia"] >= TAXA_MINIMA else "abaixo_do_limiar"

    return {
        "algoritmo": "LRS",
        "funcao": nome_funcao,
        "candidatos": resultados,
        "escolhido": escolhido,
        "status": status,
    }


def rodar_sintonia_completa(seed=20251201):
    rng = np.random.default_rng(seed)
    relatorio = {}
    parametros_finais = {}

    for nome in CONFIG.keys():
        print(f"\n=== Sintonia {nome.upper()} ===")
        r_hc = sintonizar_hc(nome, rng)
        r_lrs = sintonizar_lrs(nome, rng)

        print(f"  HC:  epsilon escolhido = {r_hc['escolhido']['epsilon']:.4f} "
              f"(percent={r_hc['escolhido']['percentual']*100:.1f}%, "
              f"taxa={r_hc['escolhido']['taxa_convergencia']*100:.0f}%, "
              f"status={r_hc['status']})")
        print(f"  LRS: sigma escolhido = {r_lrs['escolhido']['sigma']:.2f} "
              f"(taxa={r_lrs['escolhido']['taxa_convergencia']*100:.0f}%, "
              f"status={r_lrs['status']})")

        relatorio[nome] = {"HC": r_hc, "LRS": r_lrs}
        parametros_finais[nome] = {
            "epsilon_hc": float(r_hc["escolhido"]["epsilon"]),
            "percentual_hc": float(r_hc["escolhido"]["percentual"]),
            "status_hc": r_hc["status"],
            "taxa_hc": float(r_hc["escolhido"]["taxa_convergencia"]),
            "sigma_lrs": float(r_lrs["escolhido"]["sigma"]),
            "status_lrs": r_lrs["status"],
            "taxa_lrs": float(r_lrs["escolhido"]["taxa_convergencia"]),
        }

    with open("resultados/parametros_sintonizados.json", "w") as fh:
        json.dump(parametros_finais, fh, indent=2, ensure_ascii=False)

    # Salvar resumo completo da sintonia (sem listas grandes de f_finais)
    relatorio_resumo = {}
    for nome, info in relatorio.items():
        relatorio_resumo[nome] = {}
        for alg in ("HC", "LRS"):
            esc = info[alg]["escolhido"]
            cands = [{k: v for k, v in c.items() if k != "f_finais"}
                     for c in info[alg]["candidatos"]]
            relatorio_resumo[nome][alg] = {
                "status": info[alg]["status"],
                "escolhido": {k: v for k, v in esc.items() if k != "f_finais"},
                "candidatos": cands,
            }
    with open("resultados/sintonia_detalhada.json", "w") as fh:
        json.dump(relatorio_resumo, fh, indent=2, ensure_ascii=False)

    return relatorio, parametros_finais


if __name__ == "__main__":
    rodar_sintonia_completa()
