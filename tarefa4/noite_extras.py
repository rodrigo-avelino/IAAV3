"""
Master script - testes da ultima noite de execucao.

Ordem:
    1. SA v2: sigma_sintonia + T0 (~6 min)
    2. GA sigma_decrescente v2: refinamento fino (~11 min)

Tempo estimado total: ~17 min.
"""
import time
import sa_rastrigin_v2
import ga_sigma_decrescente_v2


def main():
    t0 = time.time()

    print("\n" + "=" * 70)
    print(" 1/2  SA v2: sintonia conjunta de sigma e T0")
    print("=" * 70)
    sa_rastrigin_v2.rodar()

    print("\n" + "=" * 70)
    print(" 2/2  GA sigma_decrescente v2: refinamento fino")
    print("=" * 70)
    ga_sigma_decrescente_v2.rodar()

    print(f"\n>>> TEMPO TOTAL: {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
