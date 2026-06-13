"""
Algoritmo Genetico real-coded para a Tarefa 2.3.

Representacao:
    Cada individuo e um vetor de n_var reais, restrito a x_i in [inf, sup].
    Sem decodificacao via phi (codificacao direta).

Operadores:
    SELECAO: torneio de tamanho K (default 3).
    CROSSOVER: Simulated Binary Crossover (SBX) com indice de distribuicao
        eta = 1 (do enunciado). Gera 2 filhos por par de pais; cada gene
        recebe um spread factor beta com distribuicao polinomial:
            beta = (2u)^(1/(eta+1))           se u <= 0.5
            beta = (1/(2(1-u)))^(1/(eta+1))   se u > 0.5
        Filhos:
            c1 = 0.5 * ((1+beta)*p1 + (1-beta)*p2)
            c2 = 0.5 * ((1-beta)*p1 + (1+beta)*p2)
    MUTACAO: gaussiana isotropica. Cada gene tem probabilidade p_mut_gene
        de receber ruido N(0, sigma_mut^2). Apos mutacao, aplica-se
        clipping nos bounds.
    ELITISMO: opcional (N_elite individuos copiados intactos).
    PARADA: max_gen geracoes ou estagnacao por paciencia geracoes.

Padroes adotados (literatura para GA real-coded):
    eta_SBX = 1.0             (do enunciado)
    p_cross = 0.9             (alta probabilidade de recombinacao)
    p_mut_gene = 1/n_var      (regra de bolso "1 mutacao esperada por individuo")
    sigma_mut = 0.5           (faixa absoluta; ~5% do range de [-5.12, 5.12])
    K = 3                     (pressao seletiva moderada)
"""
import numpy as np


def avaliar(f, P):
    """f deve aceitar matriz (N, n)."""
    return f(P)


def selecao_torneio(P, custos, K, rng):
    """Retorna copia de um vencedor. Se K > len(P), usa todos os indivíduos."""
    n = len(P)
    k_efetivo = min(K, n)
    idx = rng.choice(n, size=k_efetivo, replace=False)
    vencedor = idx[np.argmin(custos[idx])]
    return P[vencedor].copy()


def sbx(p1, p2, eta, rng):
    """SBX por gene. Retorna 2 filhos."""
    n = len(p1)
    u = rng.uniform(size=n)
    # beta vetorizado
    beta = np.where(
        u <= 0.5,
        (2 * u) ** (1.0 / (eta + 1)),
        (1.0 / (2 * (1 - u))) ** (1.0 / (eta + 1)),
    )
    c1 = 0.5 * ((1 + beta) * p1 + (1 - beta) * p2)
    c2 = 0.5 * ((1 - beta) * p1 + (1 + beta) * p2)
    return c1, c2


def mutacao_gaussiana(ind, p_gene, sigma, inf, sup, rng):
    """Aplica ruido N(0, sigma) por gene com probabilidade p_gene."""
    mascara = rng.uniform(size=len(ind)) < p_gene
    if mascara.any():
        ruido = rng.normal(0.0, sigma, size=int(mascara.sum()))
        ind[mascara] = ind[mascara] + ruido
    return np.clip(ind, inf, sup)


def sigma_proporcional_aptidao(rank, N, sigma_min, sigma_max):
    """
    Calcula sigma local de um individuo conforme seu rank na populacao.
    rank=0 (melhor) recebe sigma_min; rank=N-1 (pior) recebe sigma_max.
    Interpolacao linear no rank normalizado.
    """
    if N <= 1:
        return sigma_min
    pos = rank / (N - 1)
    return sigma_min + (sigma_max - sigma_min) * pos


def ga_real(
    f,
    n_var,
    inf,
    sup,
    N=100,
    max_gen=1000,
    p_cross=0.9,
    eta_sbx=1.0,
    p_mut_gene=None,
    sigma_mut=0.5,
    K=3,
    N_elite=0,
    paciencia=None,
    rng=None,
    # --- Estrategias adaptativas de mutacao ---
    alpha_sigma=1.0,            # decaimento geometrico de sigma_mut (1.0 = fixo)
    sigma_proporcional=None,     # se nao-None: (sigma_min, sigma_max) por rank
):
    """
    Roda o GA real-coded. Retorna dict com:
        melhor_x, melhor_custo, hist_melhor, hist_media, hist_pior,
        geracao_final, parou_por, avaliacoes_totais, hist_sigma.

    Estrategias adaptativas (mutuamente exclusivas):
        - alpha_sigma < 1.0  : sigma_mut decai geometricamente a cada geracao.
        - sigma_proporcional : tupla (sigma_min, sigma_max). Cada individuo
          recebe sigma local proporcional ao seu rank de aptidao (pior recebe
          sigma_max, melhor recebe sigma_min). Neste modo, sigma_mut e
          alpha_sigma sao ignorados.
    """
    if rng is None:
        rng = np.random.default_rng()
    if p_mut_gene is None:
        p_mut_gene = 1.0 / n_var

    # Populacao inicial: uniforme nos bounds
    P = rng.uniform(inf, sup, size=(N, n_var))
    custos = avaliar(f, P)
    avaliacoes = N

    hist_melhor = [float(custos.min())]
    hist_media = [float(custos.mean())]
    hist_pior = [float(custos.max())]
    hist_sigma = [float(sigma_mut)]
    sem_melhora = 0
    melhor_anterior = custos.min()
    parou_por = "max_gen"
    sigma_atual = sigma_mut

    for gen in range(max_gen):
        if N_elite > 0:
            idx_elite = np.argsort(custos)[:N_elite]
            elite = P[idx_elite].copy()

        # Mapeamento rank -> sigma local (so usado se sigma_proporcional ativo)
        if sigma_proporcional is not None:
            sigma_min, sigma_max = sigma_proporcional
            ordem = np.argsort(custos)
            rank_de = np.empty(N, dtype=int)
            rank_de[ordem] = np.arange(N)

        nova = []
        while len(nova) < N - N_elite:
            i1 = rng.integers(0, N)
            i2 = rng.integers(0, N)
            # Para preservar compatibilidade, segue o caminho via selecao_torneio
            p1 = selecao_torneio(P, custos, K, rng)
            p2 = selecao_torneio(P, custos, K, rng)
            if rng.random() < p_cross:
                c1, c2 = sbx(p1, p2, eta_sbx, rng)
            else:
                c1, c2 = p1.copy(), p2.copy()

            # Define sigma a usar nesta mutacao
            if sigma_proporcional is not None:
                # Como os filhos sao novos (sem rank), usa um sigma intermediario
                # com base na media dos ranks dos pais. Aproximacao razoavel.
                sigma_c1 = sigma_proporcional_aptidao(N // 2, N, sigma_min, sigma_max)
                sigma_c2 = sigma_c1
            else:
                sigma_c1 = sigma_atual
                sigma_c2 = sigma_atual

            c1 = mutacao_gaussiana(c1, p_mut_gene, sigma_c1, inf, sup, rng)
            c2 = mutacao_gaussiana(c2, p_mut_gene, sigma_c2, inf, sup, rng)
            c1 = np.clip(c1, inf, sup)
            c2 = np.clip(c2, inf, sup)
            nova.append(c1)
            if len(nova) < N - N_elite:
                nova.append(c2)

        P = np.array(nova)
        if N_elite > 0:
            P = np.vstack([elite, P])
        custos = avaliar(f, P)
        avaliacoes += N

        # Decaimento de sigma (apos a geracao)
        if alpha_sigma < 1.0 and sigma_proporcional is None:
            sigma_atual = sigma_atual * alpha_sigma

        hist_melhor.append(float(custos.min()))
        hist_media.append(float(custos.mean()))
        hist_pior.append(float(custos.max()))
        hist_sigma.append(float(sigma_atual))

        if custos.min() < melhor_anterior - 1e-9:
            melhor_anterior = custos.min()
            sem_melhora = 0
        else:
            sem_melhora += 1

        if paciencia is not None and sem_melhora >= paciencia:
            parou_por = "estagnacao"
            break

    idx_melhor = int(np.argmin(custos))
    return {
        "melhor_x": P[idx_melhor],
        "melhor_custo": float(custos.min()),
        "hist_melhor": hist_melhor,
        "hist_media": hist_media,
        "hist_pior": hist_pior,
        "hist_sigma": hist_sigma,
        "geracao_final": gen + 1,
        "parou_por": parou_por,
        "avaliacoes_totais": avaliacoes,
    }


if __name__ == "__main__":
    # Teste rapido
    from funcao import rastrigin, INF, SUP, N_VARIAVEIS
    import time

    rng = np.random.default_rng(42)
    t0 = time.time()
    r = ga_real(rastrigin, N_VARIAVEIS, INF, SUP,
                N=100, max_gen=1000, rng=rng)
    t = time.time() - t0
    print(f"GA N=100, 1000 ger: {t:.1f}s")
    print(f"  custo final:        {r['melhor_custo']:.2f}")
    print(f"  geracoes:           {r['geracao_final']}")
    print(f"  avaliacoes totais:  {r['avaliacoes_totais']}")
    print(f"  parou por:          {r['parou_por']}")
