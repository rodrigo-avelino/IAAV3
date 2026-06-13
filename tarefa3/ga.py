"""
Algoritmo Genético para o Caixeiro Viajante 3D.

Representação cromossômica:
    Cada indivíduo é uma permutação dos índices [1, 2, ..., N], onde N é
    o número de cidades a visitar (a origem fica fixa nas extremidades da
    rota). O comprimento do cromossomo é N.

Função custo (Psi):
    Psi(x) = sum_{i=0..N-1} d(rota[i], rota[(i+1) mod (N+1)])
    onde rota = [origem, x[0], x[1], ..., x[N-1], origem].
    Isto é, o caminho fechado origem -> x[0] -> ... -> x[N-1] -> origem.
    Problema de MINIMIZAÇÃO.

Operadores:
    - Seleção por torneio (tournament size = TS).
    - Crossover de dois pontos com reparo (Order Crossover - OX):
        Sorteia dois pontos de corte. A janela [i:j] do pai_1 é copiada
        para o filho_1; os demais genes são preenchidos na ordem em que
        aparecem em pai_2, pulando os já presentes. Análogo para filho_2.
        Esta é a variação combinatorial canônica do crossover de dois pontos.
    - Mutação por troca (swap): com probabilidade p_m, sorteia duas posições
      e troca seus valores.
    - Elitismo opcional: os N_e melhores indivíduos da geração t são copiados
      diretamente para t+1 sem passar pelos operadores.

Parada:
    - max_gen gerações atingidas.
    - Critério opcional de aptidão aceitável (não há "ótimo conhecido" sem
      resolver o problema; aceita-se quando a melhor distância não muda por
      ESTAGNACAO_MAX gerações consecutivas).

Aptidão para seleção:
    Como é minimização, a aptidão do torneio é o INVERSO ou o NEGATIVO do
    custo. Aqui usamos comparação direta por custo (o menor vence).
"""
import numpy as np


def custo_rota(rota, D):
    """
    rota: permutação dos índices das cidades (sem a origem).
    D: matriz de distâncias (origem é o índice 0).

    Implementação vetorizada: monta a sequência completa [0, rota[0], ...,
    rota[-1], 0] e soma D[seq[:-1], seq[1:]] em uma única operação numpy.
    """
    seq = np.empty(len(rota) + 2, dtype=np.int64)
    seq[0] = 0
    seq[1:-1] = rota
    seq[-1] = 0
    return float(D[seq[:-1], seq[1:]].sum())


def avaliar_populacao(P, D):
    """
    Vetoriza a avaliação de N indivíduos. Em vez de chamar custo_rota N
    vezes em loop Python, monta uma matriz (N, n_cidades+2) com origem nas
    pontas e usa indexação numpy 2D em D.
    """
    N, n = P.shape
    seq = np.empty((N, n + 2), dtype=np.int64)
    seq[:, 0] = 0
    seq[:, 1:-1] = P
    seq[:, -1] = 0
    # D[seq[:, :-1], seq[:, 1:]] é matriz (N, n+1) com cada distância parcial
    return D[seq[:, :-1], seq[:, 1:]].sum(axis=1)


# ----------------------------- Operadores -----------------------------

def selecao_torneio(P, custos, tamanho_torneio, rng):
    """Retorna uma cópia de um indivíduo vencedor por torneio."""
    N = len(P)
    candidatos = rng.choice(N, size=tamanho_torneio, replace=False)
    vencedor = candidatos[np.argmin(custos[candidatos])]
    return P[vencedor].copy()


def crossover_dois_pontos_ox(pai1, pai2, rng):
    """
    Crossover de dois pontos com reparo no estilo Order Crossover (OX).

    1. Sorteia i < j em [0, n).
    2. Filho1 herda pai1[i:j+1] na mesma posição (segmento preservado).
    3. Restantes de filho1 são preenchidos com os genes de pai2 na ordem
       circular a partir de j+1, pulando os já presentes no segmento.
    4. Idem para filho2 (trocando os papéis).

    Implementação vetorizada com máscara booleana — evita o loop Python
    com set.add() que era o gargalo da versão anterior.
    """
    n = len(pai1)
    i, j = sorted(rng.choice(n, size=2, replace=False))
    if i == j:
        j = (j + 1) % n
        i, j = sorted([i, j])

    # Posições a preencher no filho, em ordem circular começando em j+1.
    # Comprimento: n - (j+1 - i) = n - tamanho_do_segmento.
    posicoes_alvo = np.concatenate([np.arange(j + 1, n), np.arange(0, i)])

    def ox(pai_doador_segmento, pai_doador_ordem):
        filho = np.empty(n, dtype=np.int64)
        seg = pai_doador_segmento[i:j + 1]
        filho[i:j + 1] = seg

        # Máscara dos alelos já usados (alelos são 1..n, então tamanho n+1)
        usados = np.zeros(n + 2, dtype=bool)
        usados[seg] = True

        # Leitura circular do segundo pai a partir de j+1
        ordem_b = np.concatenate([pai_doador_ordem[j + 1:],
                                  pai_doador_ordem[:j + 1]])
        restantes = ordem_b[~usados[ordem_b]]

        filho[posicoes_alvo] = restantes
        return filho

    return ox(pai1, pai2), ox(pai2, pai1)


def mutacao_swap(individuo, prob, rng):
    if rng.random() < prob:
        i, j = rng.choice(len(individuo), size=2, replace=False)
        individuo[i], individuo[j] = individuo[j], individuo[i]
    return individuo


# ----------------------------- GA principal -----------------------------

def ga_caixeiro(D, n_cidades, N=100, max_gen=500, p_mut=0.01,
                tamanho_torneio=3, N_elite=0, paciencia=None, rng=None):
    """
    Roda o GA. Retorna dicionário com:
        melhor_x, melhor_custo, hist_melhor, hist_media, hist_pior,
        geracao_final, parou_por.
    """
    if rng is None:
        rng = np.random.default_rng()

    # População inicial: N permutações aleatórias das N_cidades cidades (índices 1..N_cidades)
    P = np.array([rng.permutation(np.arange(1, n_cidades + 1)) for _ in range(N)])
    custos = avaliar_populacao(P, D)

    hist_melhor = [float(custos.min())]
    hist_media = [float(custos.mean())]
    hist_pior = [float(custos.max())]
    sem_melhora = 0
    melhor_anterior = custos.min()
    parou_por = "max_gen"

    for gen in range(max_gen):
        # Elitismo: separa os N_elite melhores
        if N_elite > 0:
            idx_elite = np.argsort(custos)[:N_elite]
            elite = P[idx_elite].copy()

        # Forma nova população via torneio + crossover + mutação
        nova = []
        while len(nova) < N - N_elite:
            p1 = selecao_torneio(P, custos, tamanho_torneio, rng)
            p2 = selecao_torneio(P, custos, tamanho_torneio, rng)
            f1, f2 = crossover_dois_pontos_ox(p1, p2, rng)
            f1 = mutacao_swap(f1, p_mut, rng)
            f2 = mutacao_swap(f2, p_mut, rng)
            nova.append(f1)
            if len(nova) < N - N_elite:
                nova.append(f2)

        P = np.array(nova)
        if N_elite > 0:
            P = np.vstack([elite, P])
        custos = avaliar_populacao(P, D)

        hist_melhor.append(float(custos.min()))
        hist_media.append(float(custos.mean()))
        hist_pior.append(float(custos.max()))

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
        "geracao_final": gen + 1,
        "parou_por": parou_por,
        "populacao_final": P,
        "custos_finais": custos,
    }
