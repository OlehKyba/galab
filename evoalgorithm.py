from pressure_stats import PressureStats
from noise_stats import NoiseStats
from selection_diff_stats import SelectionDiffStats
from reproduction_stats import ReproductionStats
from run import Run
from functions import *
from constants import *


class EvoAlgorithm:
    def __init__(
        self,
        initial_population: Population,
        selection_function,
        fitness_function,
        optimal,
    ):
        self.population: Population = initial_population
        self.selection_function = selection_function
        self.iteration = 0
        self.pressure_stats = PressureStats()
        self.reproduction_stats = ReproductionStats()
        self.selection_diff_stats = SelectionDiffStats()
        best_genotypes = self.population.get_best_genotypes()
        self.pressure_stats.num_of_best.append(
            self.population.get_chromosomes_copies_counts(best_genotypes)
        )
        self.pressure_stats.f_best.append(self.population.get_max_fitness())
        self.fitness_function = fitness_function
        self.optimal = optimal

    def run(self, run, folder_name, iterations_to_plot):
        self.iteration = 0
        avg_fitness_list = [self.population.get_mean_fitness()]
        std_fitness_list = [self.population.get_fitness_std()]
        stop = 1000 if "Disruptive" in self.selection_function.__class__.__name__ else G
        convergent = self.population.estimate_convergence()

        while not convergent and self.iteration < stop:
            if self.iteration < iterations_to_plot and run < iterations_to_plot:
                self.fitness_function.draw_histograms(
                    population=self.population,
                    folder_name=folder_name,
                    func_name=self.selection_function.__class__.__name__,
                    run=run + 1,
                    iteration=self.iteration + 1,
                )

            best_genotypes = self.population.get_best_genotypes()
            f = avg_fitness_list[self.iteration]
            self.population = self.selection_function.select(self.population)
            keys_after_selection = self.population.get_keys_list()
            selected_chromosome_keys = set(keys_after_selection)
            f_parents_pool = self.population.get_mean_fitness()
            self.population.crossover(self.fitness_function)
            self.population.mutate(self.fitness_function)
            f_std = self.population.get_fitness_std()
            std_fitness_list.append(f_std)
            fs = self.population.get_mean_fitness()
            avg_fitness_list.append(fs)
            self.selection_diff_stats.s_list.append(f_parents_pool - f)
            num_of_best = self.population.get_chromosomes_copies_counts(best_genotypes)
            self.reproduction_stats.rr_list.append(
                len(selected_chromosome_keys) / N
            )
            self.reproduction_stats.best_rr_list.append(
                num_of_best / len(self.population.chromosomes)
            )
            self.pressure_stats.intensities.append(
                PressureStats.calculate_intensity(
                    f_parents_pool, f, std_fitness_list[self.iteration]
                )
            )
            self.pressure_stats.f_best.append(self.population.get_max_fitness())
            self.pressure_stats.num_of_best.append(num_of_best)
            self.iteration += 1
            self.pressure_stats.grs.append(
                PressureStats.calculate_growth_rate(
                    self.pressure_stats.num_of_best[self.iteration],
                    self.pressure_stats.num_of_best[self.iteration - 1],
                    self.pressure_stats.f_best[self.iteration],
                    self.pressure_stats.f_best[self.iteration - 1],
                )
            )
            if num_of_best >= N / 2 and self.pressure_stats.grl is None:
                self.pressure_stats.grli = self.iteration
                self.pressure_stats.grl = self.pressure_stats.grs[-1]
            convergent = self.population.estimate_convergence()
            self.population.override_chromosome_keys()

        if convergent:
            self.pressure_stats.NI = self.iteration

        if run < iterations_to_plot:
            self.fitness_function.draw_histograms(
                population=self.population,
                folder_name=folder_name,
                func_name=self.selection_function.__class__.__name__,
                run=run + 1,
                iteration=self.iteration + 1,
            )

        ns = NoiseStats() if self.fitness_function.__class__.__name__ == "Fconst" else None
        self.pressure_stats.takeover_time = self.iteration
        self.pressure_stats.f_found = self.population.get_max_fitness()
        self.pressure_stats.f_avg = self.population.get_mean_fitness()
        self.pressure_stats.calculate()
        self.reproduction_stats.calculate()
        self.selection_diff_stats.calculate()
        is_successful = self.check_success() if convergent else False

        if is_successful and ns:
            ns.NI = self.iteration
            ns.conv_to = self.population.chromosomes[0].code[0]

        return Run(
            avg_fitness_list,
            std_fitness_list,
            self.pressure_stats,
            self.reproduction_stats,
            self.selection_diff_stats,
            ns,
            is_successful,
        )

    def check_success(self):
        ff_name = self.fitness_function.__class__.__name__
        if ff_name == "FH" or ff_name == "FHD":
            optimal_chromosome = list(self.optimal.code)
            optimal_chromosomes = self.population.get_chromosomes_copies_count(optimal_chromosome)
            return optimal_chromosomes == N
        elif ff_name == "Fconst":
            return self.population.is_identical
        else:
            return any(
                [
                    self.fitness_function.check_chromosome_success(p)
                    for p in self.population.chromosomes
                ]
            )

    @staticmethod
    def calculate_noise(sf) -> Run:
        pop = Fconst().generate_population(N, 100, 0, 0)
        population = Population(pop.chromosomes.copy(), pop.p_m, pop.c_m)
        iteration = 0
        stop = 1000 if "Disruptive" in sf.__class__.__name__ else G
        reproduction_stats = ReproductionStats()
        is_successful = False

        while not population.estimate_convergence() and iteration < stop:
            keys_before_selection = population.get_keys_list()
            best_genotypes = population.get_best_genotypes()

            population = sf.select(population)

            keys_after_selection = population.get_keys_list()
            not_selected_chromosomes = set(keys_before_selection) - set(
                keys_after_selection
            )
            num_of_best = population.get_chromosomes_copies_counts(best_genotypes)

            reproduction_stats.rr_list.append(
                1 - (len(not_selected_chromosomes) / N)
            )
            reproduction_stats.best_rr_list.append(
                num_of_best / len(population.chromosomes)
            )

            iteration += 1

        ns = NoiseStats()
        reproduction_stats.calculate()

        if population.estimate_convergence():
            is_successful = True
            ns.NI = iteration
            ns.conv_to = population.chromosomes[0].code[0]

        return Run(
            reproduction_stats=reproduction_stats,
            noise_stats=ns,
            is_successful=is_successful,
        )
