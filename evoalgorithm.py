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

        best_chromosomes = self.population.get_best_chromosomes()
        self.pressure_stats.num_of_best.append(
            self.population.get_chromosomes_copies_count(best_chromosomes)
        )

        self.pressure_stats.f_best.append(self.population.get_max_fitness())
        self.fitness_function = fitness_function
        self.optimal = optimal

    def run(self, run, folder_name, encoding, iterations_to_plot):
        self.iteration = 0
        avg_fitness_list = [self.population.get_mean_fitness()]
        std_fitness_list = [self.population.get_fitness_std()]
        stop = 1000 if "Disruptive" in self.selection_function.__class__.__name__ else G
        convergent = self.population.estimate_convergence()
        optimal_counts = [self.population.get_chromosomes_copies_count(self.optimal)]

        while not convergent and self.iteration < stop:
            if self.iteration < iterations_to_plot and run < iterations_to_plot:
                self.fitness_function.draw_histograms(
                    population=self.population,
                    folder_name=folder_name,
                    encoding=encoding,
                    func_name=str(self.selection_function),
                    run=run + 1,
                    iteration=self.iteration + 1,
                )

            best_chromosomes = self.population.get_best_chromosomes()
            f = avg_fitness_list[self.iteration]
            self.population = self.selection_function.select(self.population)
            keys_after_selection = self.population.get_keys_list()
            selected_chromosome_keys = set(keys_after_selection)
            f_parents_pool = self.population.get_mean_fitness()
            self.population.crossover(self.fitness_function, encoding)
            self.population.mutate(self.fitness_function, encoding)
            f_std = self.population.get_fitness_std()
            std_fitness_list.append(f_std)
            fs = self.population.get_mean_fitness()
            avg_fitness_list.append(fs)
            optimal_counts.append(self.population.get_chromosomes_copies_count(self.optimal))
            self.selection_diff_stats.s_list.append(f_parents_pool - f)
            num_of_best = self.population.get_chromosomes_copies_count(best_chromosomes)
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
                encoding=encoding,
                func_name=str(self.selection_function),
                run=run + 1,
                iteration=self.iteration + 1,
            )

        self.pressure_stats.takeover_time = self.iteration
        self.pressure_stats.f_found = self.population.get_max_fitness()
        self.pressure_stats.f_avg = self.population.get_mean_fitness()
        self.pressure_stats.calculate()
        self.reproduction_stats.calculate()
        self.selection_diff_stats.calculate()
        is_successful = self.check_success(encoding) if convergent else False

        return Run(
            avg_fitness_list,
            std_fitness_list,
            self.pressure_stats,
            self.reproduction_stats,
            self.selection_diff_stats,
            None,
            is_successful,
            optimal_count=optimal_counts,
        )

    def check_success(self, encoding):
        ff_name = self.fitness_function.__class__.__name__
        if ff_name == "FHD":
            optimal_chromosomes = self.population.get_chromosomes_copies_count(self.optimal)
            if self.population.p_m:
                return optimal_chromosomes >= 0.9 * N
            else:
                return optimal_chromosomes == N
        elif ff_name == "Fconst":
            return True
        else:
            return any(
                [
                    self.fitness_function.check_chromosome_success(p, encoding)
                    for p in self.population.chromosomes
                ]
            )
