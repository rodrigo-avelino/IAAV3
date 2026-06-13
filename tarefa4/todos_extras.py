"""
Master script - roda TODOS os experimentos extras da Tarefa 4 em sequencia.

Ordem:
    1. Expansao do grid (N x sigma_mut) - ja existente em sintonia_expandida.py
    2. SA no Rastrigin (sa_rastrigin.py)
    3. GA sigma_mut decrescente (ga_sigma_decrescente.py)
    4. GA mutacao proporcional a aptidao (ga_proporcional.py)

Tempo estimado total: ~60-70 min.

Uso:
    python3 todos_extras.py
"""
import time
import sintonia_expandida
import sa_rastrigin
import ga_sigma_decrescente
import ga_proporcional


def main():
    t0 = time.time()

    print("\n" + "=" * 70)
    print(" 1/4  EXPANSAO DO GRID (N x sigma_mut)")
    print("=" * 70)
    sintonia_expandida.rodar()

    print("\n" + "=" * 70)
    print(" 2/4  TEMPERA SIMULADA (SA) NO RASTRIGIN")
    print("=" * 70)
    sa_rastrigin.rodar()

    print("\n" + "=" * 70)
    print(" 3/4  GA COM SIGMA DECRESCENTE")
    print("=" * 70)
    ga_sigma_decrescente.rodar()

    print("\n" + "=" * 70)
    print(" 4/4  GA COM MUTACAO PROPORCIONAL A APTIDAO")
    print("=" * 70)
    ga_proporcional.rodar()

    print(f"\n>>> TEMPO TOTAL: {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
