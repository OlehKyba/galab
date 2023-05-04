import random
from statistics import mean

import numpy as np

from chromosome import Chromosome


class Population:
    def __init__(self, chromosomes: list[Chromosome], p_m, c_m):
        self.chromosomes = chromosomes
        self.fitness_list: list[float] = [chromosome.fitness for chromosome in self.chromosomes]
        self.genotypes_list = [list(x.code) for x in self.chromosomes]
        self.p_m = p_m
        self.c_m = c_m

    def estimate_convergence(self):
        if self.p_m == 0:
            return self.is_identical
        else:
            return self.is_homogeneous(percentage=99)

    def is_homogeneous(self, percentage: int) -> bool:
        assert percentage < 100, (
            "According to formula: (unique / total) * 100 <= 100 - percentage, "
            "we can't have percentage >= 100!"
        )
        chromosomes = ["".join(map(str, genotype)) for genotype in self.genotypes_list]
        total = len(chromosomes)
        unique = len(set(chromosomes))
        return (unique / total) * 100 <= 100 - percentage

    @property
    def is_identical(self) -> bool:
        genotypes = {"".join(map(str, genotype)) for genotype in self.genotypes_list}
        return len(genotypes) == 1

    def crossover(self, fitness_function):
        if self.c_m == 0:
            return

        next_chromosomes = []

        chromosomes = self.chromosomes.copy()

        def pop_chromosome():
            index = random.randrange(0, len(chromosomes))
            return chromosomes.pop(index)

        next_key = 0

        while len(chromosomes) > 0:
            parent1 = pop_chromosome()
            parent2 = pop_chromosome()

            crossover_point = int(random.random() * len(parent1.code))

            child_code1 = [*parent1.code[:crossover_point], *parent2.code[crossover_point:]]
            child_chromosome1 = Chromosome(
                child_code1,
                fitness_function.estimate(child_code1),
                next_key + 1
            )
            
            child_code2 = [*parent2.code[:crossover_point], *parent1.code[crossover_point:]]
            child_chromosome2 = Chromosome(
                child_code2,
                fitness_function.estimate(child_code2),
                next_key + 2
            )

            next_chromosomes.append(child_chromosome1)
            next_chromosomes.append(child_chromosome2)

            next_key += 2

        self.update_chromosomes(next_chromosomes)

    def mutate(self, fitness_function):
        if self.p_m == 0:
            return
        for chromosome in self.chromosomes:
            for i in range(0, len(chromosome.code)):
                if random.random() < self.p_m:
                    chromosome.code[i] = int(not chromosome.code[i])
                    chromosome.fitness = fitness_function.estimate(chromosome.code)
        self.update()

    def get_mean_fitness(self):
        return mean(self.fitness_list)

    def get_max_fitness(self):
        return max(self.fitness_list)

    def get_best_genotypes(self) -> list[list[int]]:
        max_fitness = self.get_max_fitness()
        best_genotypes = [
            self.genotypes_list[index]
            for index, fitness_value in enumerate(self.fitness_list)
            if fitness_value == max_fitness
        ]
        return list(np.unique(best_genotypes, axis=0))

    def get_chromosomes_copies_counts(self, genotypes: list[list[int]]) -> int:
        all_genotypes = [''.join(map(str, genotype)) for genotype in self.genotypes_list]
        unique_genotypes = {''.join(map(str, genotype)) for genotype in genotypes}

        copies_count = 0
        for genotype in unique_genotypes:
            copies_count += all_genotypes.count(genotype)
        return copies_count

    def get_fitness_std(self):
        return np.std(self.fitness_list)

    def get_keys_list(self):
        return [chromosome.key for chromosome in self.chromosomes]

    def get_chromosomes_copies_count(self, genotype_copy):
        genotype_copy = ''.join(map(str, genotype_copy))
        genotypes = [''.join(map(str, genotype)) for genotype in self.genotypes_list]
        return genotypes.count(genotype_copy)

    def update(self):
        self.fitness_list = [chromosome.fitness for chromosome in self.chromosomes]
        self.genotypes_list = [list(x.code) for x in self.chromosomes]

    def update_chromosomes(self, chromosomes):
        self.chromosomes = chromosomes
        self.update()

    def override_chromosome_keys(self):
        for index, chromosome in enumerate(self.chromosomes):
            self.chromosomes[index] = chromosome.clone(key=index)


def population_factory(fitness_function, n, l, p_m, c_m, i) -> Population:
    chromosomes = [fitness_function.generate_optimal(l)]
    start = len(chromosomes)

    for j in range(start, n):
        code = np.random.binomial(n=1, p=0.5, size=l)
        fitness = fitness_function.estimate(code)
        chromosomes.append(Chromosome(code, fitness, j + 1))

    return Population(chromosomes, p_m, c_m)
