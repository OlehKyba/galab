env = "test"

# Runs: amount of evolutionary algorithms to probe.
MAX_RUNS = 5 if env == "test" else 100

# Genotype: amount of genes / codec.
N = 100
ENCODING = 'gray'

# Termination condition: maximum amount of iterations.
G = 10000000

# Selection: exponential parameter for Rank selection.
C = 0.979

# Operators: rate for Dense mutation / rate for Single-point crossover.
P_M = 0.0005  # depends on other criterias
P_C = 1  # 0 or 1
P_M_DICT = {
    # (N, L): probability
    (100, 10): 0.0001,
    (100, 100): 0.00001,
}

# Convergence condition:
SIGMA = DELTA = 0.01
