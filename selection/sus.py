import random

from numpy import random

from chromosome import Chromosome
from population import Population
from selection.utils import create_wheel, binary_search_by_wheel


def basic_sus(wheel: list[tuple[tuple[float, float], Chromosome]]) -> list[Chromosome]:
    arrows = len(wheel)
    arrow_step = 1 / arrows
    arrow_offset = random.random()
    left = 0
    wheel_point = arrow_offset
    parents_pool: list[Chromosome] = []

    for _ in range(arrows):
        left, chromosome = binary_search_by_wheel(wheel, wheel_point, left)
        parents_pool.append(chromosome)

        wheel_point += arrow_step
        if wheel_point > 1:
            wheel_point -= 1
            left = 0

    return parents_pool


class RankExponentialSUS:
    def __init__(self, c: float = 0.95):
        self.c = c

    def exponential_sus(self, population: Population):
        size = len(population.chromosomes)

        probabilities_with_chromosomes = [
            (self.scale(size, size - i), population.chromosomes[i])
            for i in range(size)
        ]
        wheel = create_wheel(probabilities_with_chromosomes)
        mating_pool = basic_sus(wheel)
        population.update_chromosomes(mating_pool)

        return population

    def select(self, population: Population):
        chromosomes = population.chromosomes.copy()
        random.shuffle(chromosomes)
        chromosomes = self.sort(chromosomes)
        population.update_chromosomes(chromosomes)
        return self.exponential_sus(population)

    def scale(self, size: int, rank: int) -> float:
        return ((self.c - 1) / (pow(self.c, size) - 1)) * pow(self.c, size - rank)

    def sort(self, chromosomes):
        return sorted(chromosomes.copy(), key=lambda chromosome: chromosome.fitness, reverse=True)

    def __str__(self):
        return f"RankExponentialSUS[c={self.c}]"
