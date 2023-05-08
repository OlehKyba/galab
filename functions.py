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
    encoding: str | None,
    func_name: str,
    run: int,
    iteration: int,
) -> None:
    dir_path = f"{folder_name}/{encoding}" if encoding else folder_name
    dir_path += f"/{N}/{func_name}/{run}/{iteration}"
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
    encoding: str | None,
    func_name: str,
    run: int,
    iteration: int,
) -> None:
    dir_path = f"{folder_name}/{encoding}" if encoding else folder_name
    dir_path += f"/{N}/{func_name}/{run}/{iteration}"
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    phenotypes = [
        decode(chromosome.code, a, b, len(chromosome.code), encoding)
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
    encoding: str | None,
    func_name: str,
    run: int,
    iteration: int,
) -> None:
    dir_path = f"{folder_name}/{encoding}" if encoding else folder_name
    dir_path += f"/{N}/{func_name}/{run}/{iteration}"
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


class PopulationGeneratorMixin:
    population_seeds: tuple[int, ...] | None

    def generate_population(self, n: int, l: int, p_m: float, c_m: float, i: int, encoding: str) -> Population:
        seed = self.population_seeds[i] if self.population_seeds else None
        return population_factory(
            fitness_function=self,
            n=n,
            l=l,
            p_m=p_m,
            c_m=c_m,
            i=i,
            encoding=encoding,
            seed=seed,
        )


class BinaryFuncMixin:
    IS_BINARY: bool = True

    def draw_histograms(
        self,
        population: Population,
        folder_name: str,
        encoding: str | None,
        func_name: str,
        run: int,
        iteration: int,
    ) -> None:
        _draw_count_ones_in_genotype_histogram(
            population, folder_name, encoding, func_name, run, iteration
        )


class NotBinaryFuncMixin:
    a: float
    b: float

    IS_BINARY: bool = False

    def draw_histograms(
        self,
        population: Population,
        folder_name: str,
        encoding: str | None,
        func_name: str,
        run: int,
        iteration: int,
    ) -> None:
        assert encoding, "Encoding can't be None for not binary function!"
        _draw_count_ones_in_genotype_histogram(
            population, folder_name, encoding, func_name, run, iteration
        )
        _draw_fitness_histogram(population, folder_name, encoding, func_name, run, iteration)
        _draw_phenotype_histogram(
            self.a, self.b, population, folder_name, encoding, func_name, run, iteration
        )


class FHD(PopulationGeneratorMixin, BinaryFuncMixin):
    def __init__(self, delta: float, population_seeds: tuple[int, ...] | None = None):
        self.delta = delta
        self.population_seeds = population_seeds

    def get_genotype_value(self, chromosome_code):
        return np.count_nonzero(chromosome_code)

    def estimate(self, chromosome, encoding):
        k = len(chromosome) - np.count_nonzero(chromosome)
        return (len(chromosome) - k) + k * self.delta

    def generate_optimal(self, length, encoding):
        genotype = np.zeros((length,), dtype=int)
        return Chromosome(genotype, self.estimate(genotype, encoding))

    def get_optimal(self, n, l, p_m, c_m, i, encoding):
        return self.generate_optimal(l, encoding)


class Fconst(BinaryFuncMixin):
    def __init__(self, population_seeds: tuple[int, ...] | None = None):
        self.population_seeds = population_seeds

    def estimate(self, chromosome_code, encoding):
        return len(chromosome_code)

    def generate_optimal(self, length: int, encoding: str):
        return Chromosome(np.zeros((length,), dtype=int).tolist(), length)

    def get_optimal(self, n, l, p_m, c_m, i, encoding):
        return self.generate_optimal(l, encoding)

    def generate_population(self, n: int, l: int, p_m: float, c_m: float, i: int, encoding: str) -> Population:
        seed = self.population_seeds[i] if self.population_seeds else None
        return population_factory(
            fitness_function=self,
            n=n,
            l=l,
            p_m=p_m,
            c_m=c_m,
            i=i,
            encoding=encoding,
            seed=seed,
            is_add_optimal=False,
        )


class Fx2(PopulationGeneratorMixin, NotBinaryFuncMixin):
    def __init__(self, a: float, b: float, population_seeds: tuple[int, ...] | None = None):
        self.a = a
        self.b = b
        self.extremum_x = b
        self.extremum_y = math.pow(b, 2)
        self.population_seeds = population_seeds

    def score(self, x: float):
        return math.pow(x, 2)

    def estimate(self, chromosome_code, encoding):
        x = decode(chromosome_code, self.a, self.b, len(chromosome_code), encoding)
        return self.score(x)

    def generate_optimal(self, length, encoding):
        coding = encode(self.extremum_x, self.a, self.b, length, encoding)
        return Chromosome(coding, self.extremum_y)

    def get_optimal(self, n, l, p_m, c_m, i, encoding):
        return self.generate_optimal(l, encoding)

    def check_chromosome_success(self, chromosome: Chromosome, encoding: str):
        x = decode(chromosome.code, self.a, self.b, len(chromosome.code), encoding)
        y = self.score(x) 
        return abs(self.extremum_y - y) <= DELTA and abs(self.extremum_x - x) <= SIGMA


class F5122subx2(PopulationGeneratorMixin, NotBinaryFuncMixin):
    def __init__(self, a: float, b: float, population_seeds: int | None = None):
        self.a = a
        self.b = b
        self.extremum_x = 0 
        self.extremum_y = math.pow(5.12, 2)
        self.population_seeds = population_seeds

    def score(self, x: float):
        return math.pow(5.12, 2) - math.pow(x, 2)

    def estimate(self, chromosome_code, encoding):
        x = decode(chromosome_code, self.a, self.b, len(chromosome_code), encoding)
        return math.pow(5.12, 2) - math.pow(x, 2)

    def get_optimal(self, n, l, p_m, c_m, i, encoding):
        return self.generate_optimal(l, encoding)

    def generate_optimal(self, length, encoding):
        coding = encode(self.extremum_x, self.a, self.b, length, encoding)
        return Chromosome(coding, self.extremum_y)

    def check_chromosome_success(self, chromosome: Chromosome, encoding: str):
        x = decode(chromosome.code, self.a, self.b, len(chromosome.code), encoding)
        y = self.score(x) 
        return abs(self.extremum_y - y) <= DELTA and abs(self.extremum_x - x) <= SIGMA


class Fecx(PopulationGeneratorMixin, NotBinaryFuncMixin):
    def __init__(self, a: float, b: float, c: float, population_seeds: int | None = None):
        self.a = a
        self.b = b
        self.c = c
        self.extremum_x = b
        self.extremum_y = math.exp(c * b)
        self.population_seeds = population_seeds

    def score(self, x: float):
        return math.exp(self.c * x)

    def estimate(self, chromosome_code, encoding):
        x = decode(chromosome_code, self.a, self.b, len(chromosome_code), encoding)
        return self.score(x)

    def generate_optimal(self, length, encoding):
        coding = encode(self.extremum_x, self.a, self.b, length, encoding)
        return Chromosome(coding, self.extremum_y)

    def get_optimal(self, n, l, p_m, c_m, i, encoding):
        return self.generate_optimal(l, encoding)

    def check_chromosome_success(self, chromosome: Chromosome, encoding: str):
        x = decode(chromosome.code, self.a, self.b, len(chromosome.code), encoding)
        y = self.score(x) 
        return abs(self.extremum_y - y) <= DELTA and abs(self.extremum_x - x) <= SIGMA
