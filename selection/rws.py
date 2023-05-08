import random

import numpy as np

from population import Population


class RankExponentialRWS:
    def __init__(self, c: float = 0.95):
        self.c = c

    def exponential_rws(self, population: Population):
        size = len(population.fitness_list)

        probabilities = [self.scale(size, rank) for rank in range(size, 0, -1)]
        if sum(probabilities) == 0:
            return population

        chromosomes = np.random.choice(
            population.chromosomes,
            len(population.chromosomes),
            p=probabilities,
        ).tolist()
        population.update_chromosomes(chromosomes)

        return population

    def select(self, population: Population):
        chromosomes = population.chromosomes.copy()
        random.shuffle(chromosomes)
        chromosomes = self.sort(chromosomes)
        population.update_chromosomes(chromosomes)
        return self.exponential_rws(population)

    def scale(self, size: int, rank: int):
        return ((self.c - 1) / (pow(self.c, size) - 1)) * pow(self.c, size - rank)

    def sort(self, chromosomes):
        return sorted(chromosomes.copy(), key=lambda chromosome: chromosome.fitness, reverse=True)

    def __str__(self):
        return f"RankExponentialRWS[c={self.c}]"
