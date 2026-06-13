# IAAV3 — Meta-Heurísticas Estocásticas em Problemas de Otimização

Implementações e análise comparativa de algoritmos de otimização estocástica aplicados a problemas de otimização contínua, satisfação de restrições e otimização combinatória. Trabalho desenvolvido como AV3 da disciplina de Inteligência Artificial Computacional (Universidade de Fortaleza).

## Estrutura do repositório

```
IAAV3/
├── tarefa1/    Otimização de seis funções multimodais (HC, LRS, GRS)
├── tarefa2/    Problema das Oito Rainhas (Têmpera Simulada)
├── tarefa3/    Caixeiro Viajante 3D (Algoritmo Genético permutacional)
├── tarefa4/    Função de Rastrigin n=50 (LRS, SA, GA real-coded)
└── relatorio/  Relatório IEEE em LaTeX
```

Cada pasta `tarefa*/` contém os scripts de execução, sintonia e geração de figuras, além de um subdiretório `resultados/` com os CSVs de estatísticas e PNGs das figuras do relatório.

## Algoritmos implementados

| Tarefa | Problema                           | Métodos avaliados                                                              |
|--------|------------------------------------|--------------------------------------------------------------------------------|
| 1      | Otimização de funções multimodais  | Hill Climbing, Local Random Search, Global Random Search                       |
| 2      | Oito Rainhas                       | Têmpera Simulada                                                               |
| 3      | Caixeiro Viajante 3D (160 cidades) | Algoritmo Genético permutacional (OX + swap)                                   |
| 4      | Rastrigin em 50 dimensões          | LRS, SA, GA com sigma fixo, GA com sigma decrescente, GA proporcional          |

## Reprodução dos experimentos

### Requisitos

- Python 3.10 ou superior
- `numpy`, `matplotlib`

```
pip install numpy matplotlib
```

### Execução

Cada tarefa pode ser executada de forma independente a partir do seu diretório:

```
cd tarefa<N>
python3 sintonia.py            # sintonia de hiperparâmetros
python3 avaliacao_final.py     # avaliação com a configuração vencedora
python3 graficos.py            # geração das figuras
```

Os scripts específicos variam ligeiramente entre as tarefas. Cada `resultados/` contém os CSVs com estatísticas descritivas (média, desvio-padrão, mínimo, máximo, mediana) e os PNGs das figuras utilizadas no relatório.

### Observação sobre arquivos de dados brutos

Os arquivos JSON com históricos completos de execução não são versionados (centenas de MB no total) e podem ser regerados pela reexecução dos scripts. Os CSVs com estatísticas consolidadas e os PNGs das figuras ficam versionados em cada `resultados/`.

## Principais resultados

- **Funções multimodais**: nenhuma das estratégias elementares supera o limiar de 50% de convergência em mais da metade das funções investigadas, com perfis de aplicabilidade distintos por topologia.
- **Oito Rainhas**: cobertura das 92 soluções únicas atingida em 769 execuções da Têmpera Simulada, com tempo de parede inferior a 10 segundos.
- **Caixeiro Viajante**: saturação do ganho marginal identificada em N = 10000 sob critério de redução marginal abaixo de 1%, com custo mínimo absoluto de 1746 unidades.
- **Rastrigin**: GA com sigma decrescente atinge média de 71 unidades, superando em fator de 6x a Busca Aleatória Local pura sob orçamento computacional idêntico.

## Autor

Rodrigo Avelino Lucas
Engenharia de Computação, Universidade de Fortaleza
rodrigo-avelino@edu.unifor.br

## Licença

Material acadêmico de uso educacional.