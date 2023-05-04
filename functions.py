import os

import numpy as np
import matplotlib.pyplot as plt

from chromosome import Chromosome
from constants import DELTA, SIGMA, N
from population import Population, population_factory
from coding import *


def _draw_fitness_histogram(
    population: Population,
    folder_name: str,
    func_name: str,
    run: int,
    iteration: int,
) -> None:
    dir_path = f"{folder_name}/{N}/{func_name}/{run}/{iteration}"
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    plt.figure()
    plt.hist(population.fitness_list, bins=100, color="green", histtype="bar", rwidth=1)

    # x-axis label
    plt.xlabel("Health")
    # frequency label
    plt.ylabel("Num of individual")
    # plot title

    plt.title("Розподіл здоров'я")
    plt.savefig(f"{dir_path}/fitness_histogram.png")
    plt.close()


def _draw_phenotype_histogram(
    a: float,
    b: float,
    population: Population,
    folder_name: str,
    func_name: str,
    run: int,
    iteration: int,
) -> None:
    dir_path = f"{folder_name}/{N}/{func_name}/{run}/{iteration}"
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    phenotypes = [
        decode(chromosome.code, a, b, len(chromosome.code))
        for chromosome in population.chromosomes
    ]

    plt.figure()
    plt.hist(phenotypes, bins=100, color="red", histtype="bar", rwidth=None)

    # x-axis label
    plt.xlabel("X")
    # frequency label
    plt.ylabel("Num of individual")
    # plot title

    plt.title("Значення X")

    plt.savefig(f"{dir_path}/phenotypes_histogram.png")
    plt.close()


def _draw_count_ones_in_genotype_histogram(
    population: Population,
    folder_name: str,
    func_name: str,
    run: int,
    iteration: int,
) -> None:
    dir_path = f"{folder_name}/{N}/{func_name}/{run}/{iteration}"
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    ones = [
        sum(chromosome.code)
        for chromosome in population.chromosomes
    ]
    bins = 100

    plt.figure()
    plt.hist(
        ones, bins, color="red", histtype="bar", rwidth=1
    )

    # x-axis label
    plt.xlabel("Ones in genotype")
    # frequency label
    plt.ylabel("Num of individual")
    # plot title
    plt.title("Кількість одиниць в хромосомі")

    plt.savefig(f"{dir_path}/ones_in_genotypes_histogram.png")
    plt.close()


class FHD:
    def __init__(self, delta: float):
        self.delta = delta

    def get_genotype_value(self, chromosome_code):
        return np.count_nonzero(chromosome_code)

    def estimate(self, chromosome):
        k = len(chromosome) - np.count_nonzero(chromosome)
        return (len(chromosome) - k) + k * self.delta

    def generate_optimal(self, length):
        genotype = np.zeros((length,), dtype=int)
        return Chromosome(genotype, self.estimate(genotype))

    def get_optimal(self, n, l, p_m, c_m, i):
        return self.generate_optimal(l)

    def generate_population(self, n, l, p_m, c_m, i):
        return population_factory(
            fitness_function=self,
            n=n,
            l=l,
            p_m=p_m,
            c_m=c_m,
            i=i,
        )

    def draw_histograms(
        self,
        population: Population,
        folder_name: str,
        func_name: str,
        run: int,
        iteration: int,
    ) -> None:
        _draw_count_ones_in_genotype_histogram(
            population, folder_name, func_name, run, iteration
        )


class Fconst:
    def estimate(self, chromosome: Chromosome):
        return len(chromosome.code)

    def generate_optimal(self, length: int):
        return [
            Chromosome(np.zeros((length,), dtype=int).tolist(), length),
            Chromosome(np.ones((length,), dtype=int).tolist(), length),
        ]

    def generate_population(self, n, l, p_m, c_m, i):
        chromosomes = self.generate_optimal(l) * int(n / 2)
        population = Population(chromosomes, p_m, c_m)
        population.override_chromosome_keys()
        return population

    def get_optimal(self, n, l, p_m, c_m, i):
        return self.generate_optimal(l)

    def draw_histograms(
        self,
        population: Population,
        folder_name: str,
        func_name: str,
        run: int,
        iteration: int,
    ) -> None:
        _draw_count_ones_in_genotype_histogram(
            population, folder_name, func_name, run, iteration
        )


class Fx2:
    def __init__(self, a: float, b: float):
        self.a = a
        self.b = b
        self.extremum_x = b
        self.extremum_y = math.pow(b, 2)

    def score(self, x: float):
        return math.pow(x, 2)

    def estimate(self, chromosome_code):
        x = decode(chromosome_code, self.a, self.b, len(chromosome_code))
        return self.score(x)

    def get_genotype_value(self, chromosome_code):
        x = decode(chromosome_code, self.a, self.b, len(chromosome_code))
        return self.score(x)

    def generate_optimal(self, length):
        coding = encode(self.extremum_x, self.a, self.b, length)
        return Chromosome(coding, self.extremum_y)

    def get_optimal(self, n, l, p_m, c_m, i):
        return self.generate_optimal(l)

    def generate_population(self, n, l, p_m, c_m, i):
        return population_factory(
            fitness_function=self,
            n=n,
            l=l,
            p_m=p_m,
            c_m=c_m,
            i=i,
        )

    def check_chromosome_success(self, chromosome: Chromosome):
        x = decode(chromosome.code, self.a, self.b, len(chromosome.code))
        y = self.score(x) 
        return abs(self.extremum_y - y) <= DELTA and abs(self.extremum_x - x) <= SIGMA

    def draw_histograms(
        self,
        population: Population,
        folder_name: str,
        func_name: str,
        run: int,
        iteration: int,
    ) -> None:
        _draw_count_ones_in_genotype_histogram(
            population, folder_name, func_name, run, iteration
        )
        _draw_fitness_histogram(population, folder_name, func_name, run, iteration)
        _draw_phenotype_histogram(
            self.a, self.b, population, folder_name, func_name, run, iteration
        )


class F5122subx2:
    def __init__(self, a: float, b: float):
        self.a = a
        self.b = b
        self.extremum_x = 0 
        self.extremum_y = math.pow(5.12, 2)

    def score(self, x: float):
        return math.pow(5.12, 2) - math.pow(x, 2)

    def estimate(self, chromosome_code):
        x = decode(chromosome_code, self.a, self.b, len(chromosome_code))
        return math.pow(5.12, 2) - math.pow(x, 2)

    def get_genotype_value(self, chromosome_code):
        x = decode(chromosome_code, self.a, self.b, len(chromosome_code))
        return x

    def get_optimal(self, n, l, p_m, c_m, i):
        return self.generate_optimal(l)

    def generate_optimal(self, length):
        coding = encode(self.extremum_x, self.a, self.b, length)
        return Chromosome(coding, self.extremum_y)

    def generate_population(self, n, l, p_m, c_m, i):
        return population_factory(
            fitness_function=self,
            n=n,
            l=l,
            p_m=p_m,
            c_m=c_m,
            i=i,
        )

    def check_chromosome_success(self, chromosome: Chromosome):
        x = decode(chromosome.code, self.a, self.b, len(chromosome.code))
        y = self.score(x) 
        return abs(self.extremum_y - y) <= DELTA and abs(self.extremum_x - x) <= SIGMA

    def draw_histograms(
        self,
        population: Population,
        folder_name: str,
        func_name: str,
        run: int,
        iteration: int,
    ) -> None:
        _draw_count_ones_in_genotype_histogram(
            population, folder_name, func_name, run, iteration
        )
        _draw_fitness_histogram(population, folder_name, func_name, run, iteration)
        _draw_phenotype_histogram(
            self.a, self.b, population, folder_name, func_name, run, iteration
        )


class Fecx:
    def __init__(self, a: float, b: float, c: float):
        self.a = a
        self.b = b
        self.c = c
        self.extremum_x = b
        self.extremum_y = math.exp(c * b)

    def score(self, x: float):
        return math.exp(self.c * x)

    def estimate(self, chromosome_code):
        x = decode(chromosome_code, self.a, self.b, len(chromosome_code))
        return self.score(x)

    def get_genotype_value(self, chromosome_code):
        x = decode(chromosome_code, self.a, self.b, len(chromosome_code))
        return self.score(x)

    def generate_optimal(self, length):
        coding = encode(self.extremum_x, self.a, self.b, length)
        return Chromosome(coding, self.extremum_y)

    def get_optimal(self, n, l, p_m, c_m, i):
        return self.generate_optimal(l)

    def generate_population(self, n, l, p_m, c_m, i):
        return population_factory(
            fitness_function=self,
            n=n,
            l=l,
            p_m=p_m,
            c_m=c_m,
            i=i,
        )

    def check_chromosome_success(self, chromosome: Chromosome):
        x = decode(chromosome.code, self.a, self.b, len(chromosome.code))
        y = self.score(x) 
        return abs(self.extremum_y - y) <= DELTA and abs(self.extremum_x - x) <= SIGMA

    def draw_histograms(
        self,
        population: Population,
        folder_name: str,
        func_name: str,
        run: int,
        iteration: int,
    ) -> None:
        _draw_count_ones_in_genotype_histogram(
            population, folder_name, func_name, run, iteration
        )
        _draw_fitness_histogram(population, folder_name, func_name, run, iteration)
        _draw_phenotype_histogram(
            self.a, self.b, population, folder_name, func_name, run, iteration
        )
