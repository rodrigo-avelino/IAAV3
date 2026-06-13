"""
Carregador de pontos para o Caixeiro Viajante 3D em grupos.

Formato esperado do CSV (CaixeiroGruposGA.csv):
    - sem cabeçalho
    - separador vírgula
    - 4 colunas: x, y, z, grupo
    - origem identificada pelo grupo 0 (ou pelo ponto de menor norma)

Após o carregamento, a origem é movida para o índice 0 do array, e as
160 cidades a visitar ocupam os índices 1..160. Os índices 1..160 são
os "alelos" usados nos cromossomos do GA: cada cromossomo é uma
permutação de [1, 2, ..., 160].
"""
import numpy as np


def carregar_csv(caminho):
    """
    Carrega CaixeiroGruposGA.csv (formato sem cabeçalho, 4 colunas).
    Retorna:
        cidades: array (N+1, 3) com a ORIGEM no índice 0
        grupos:  array (N+1,) com etiquetas (-1 para a origem)
    """
    data = np.loadtxt(caminho, delimiter=",")
    pontos = data[:, :3]
    grupos = data[:, 3].astype(int)

    # Identifica a origem: grupo 0 ou, na ausência, o ponto de menor norma
    idx_origem_grupo0 = np.where(grupos == 0)[0]
    if len(idx_origem_grupo0) > 0:
        idx_origem = int(idx_origem_grupo0[0])
    else:
        idx_origem = int(np.argmin(np.linalg.norm(pontos, axis=1)))

    # Reorganiza colocando a origem no índice 0
    ordem = np.concatenate([[idx_origem],
                            np.delete(np.arange(len(pontos)), idx_origem)])
    cidades = pontos[ordem]
    grupos_reord = grupos[ordem]
    # Marca a origem com grupo -1 (sentinela)
    grupos_reord[0] = -1

    return cidades, grupos_reord


def matriz_distancias(cidades):
    """Pré-computa matriz de distâncias euclidianas."""
    diff = cidades[:, None, :] - cidades[None, :, :]
    return np.sqrt((diff ** 2).sum(axis=2))


def preparar_problema(caminho_csv):
    """
    Carrega o CSV, monta a matriz de distâncias e devolve tudo pronto
    para alimentar o GA.

    Retorna:
        cidades, grupos, D (161x161), n_cidades (160)
    """
    cidades, grupos = carregar_csv(caminho_csv)
    D = matriz_distancias(cidades)
    n_cidades = len(cidades) - 1  # exclui a origem
    return cidades, grupos, D, n_cidades


if __name__ == "__main__":
    cidades, grupos, D, n = preparar_problema("CaixeiroGruposGA.csv")
    print(f"Total de pontos:    {len(cidades)} ({n} cidades + 1 origem)")
    print(f"Origem (indice 0):  ({cidades[0,0]:.2f}, {cidades[0,1]:.2f}, {cidades[0,2]:.2f})")
    print(f"Grupos presentes:   {sorted(set(int(g) for g in grupos))}")
    print(f"Contagem por grupo: ", end="")
    for g in sorted(set(int(g) for g in grupos)):
        n_g = (grupos == g).sum()
        print(f"{g}: {n_g}  ", end="")
    print()
    print(f"\nMatriz de distancias: {D.shape}")
    print(f"  Distancia media:        {D[D > 0].mean():.2f}")
    print(f"  Distancia maxima:       {D.max():.2f}")
    print(f"  Distancia origem-cidade:")
    print(f"    min {D[0, 1:].min():.2f}   media {D[0, 1:].mean():.2f}   max {D[0, 1:].max():.2f}")
