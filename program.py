from multiprocessing import Pool

from constants import MAX_RUNS
from run import Run
from runs_stats import RunsStats
from evoalgorithm import EvoAlgorithm
from population import Population
from excel import save_to_excel, save_noise_to_excel
from plots import *
import time


def save_run_plots(ff_name, encoding, sf_name, run, iteration):
    save_line_plot(
        ff_name,
        encoding,
        sf_name,
        run.avg_fitness_list,
        "f_avg" + str(iteration + 1),
        "f avg",
        iteration + 1,
    )
    save_line_plot(
        ff_name,
        encoding,
        sf_name,
        run.pressure_stats.f_best,
        "f_best" + str(iteration + 1),
        "f best",
        iteration + 1,
    )
    save_line_plot(
        ff_name,
        encoding,
        sf_name,
        run.std_fitness_list,
        "f_std" + str(iteration + 1),
        "f std",
        iteration + 1,
    )
    save_line_plot(
        ff_name,
        encoding,
        sf_name,
        run.pressure_stats.intensities,
        "intensity" + str(iteration + 1),
        "intensity",
        iteration + 1,
    )
    save_line_plot(
        ff_name,
        encoding,
        sf_name,
        run.selection_diff_stats.s_list,
        "selection_diff" + str(iteration + 1),
        "selection difference",
        iteration + 1,
    )
    save_lines_plot(
        ff_name,
        encoding,
        sf_name,
        [run.pressure_stats.intensities, run.selection_diff_stats.s_list],
        ["Intensity", "EvoAlgorithm diff"],
        "intensity_and_sel_diff" + str(iteration + 1),
        "Intensity + EvoAlgorithm diff",
        iteration + 1,
    )
    save_line_plot(
        ff_name,
        encoding,
        sf_name,
        run.pressure_stats.grs,
        "gr" + str(iteration + 1),
        "growth rate",
        iteration + 1,
    )
    save_lines_plot(
        ff_name,
        encoding,
        sf_name,
        [
            run.reproduction_stats.rr_list,
            [1 - rr for rr in run.reproduction_stats.rr_list],
        ],
        ["Reproduction rate", "Loss of diversity"],
        "repro_rate_and_loss_of_diversity" + str(iteration + 1),
        "Reproduction rate + Loss of diversity",
        iteration + 1,
    )
    save_line_plot(
        ff_name,
        encoding,
        sf_name,
        run.reproduction_stats.best_rr_list,
        "best_rr" + str(iteration + 1),
        "best chromosome rate",
        iteration + 1,
    )
    save_line_plot(
        ff_name,
        encoding,
        sf_name,
        run.optimal_count,
        "optimal_count" + str(iteration + 1),
        "Optimal count",
        iteration + 1,
    )


def main(fitness_function, selection_functions: list, file_name: str, encoding: str, pool: Pool, *args):
    p_start = time.time()
    runs_dict = {}
    ff_name = fitness_function.__class__.__name__
    print(f"Program {file_name} ({encoding}) started (in sec.): {p_start}")

    for selection_function in selection_functions:
        runs_dict[str(selection_function)] = RunsStats()

    all_runs_args = [
        (
            selection_functions,
            fitness_function,
            file_name,
            encoding,
            ff_name,
            fitness_function.generate_population(*args, encoding=encoding, i=i),
            i,
            args,
        )
        for i in range(0, MAX_RUNS)
    ]

    for results in pool.starmap(run_algo, all_runs_args):
        for sf_name, run in results:
            runs_dict[sf_name].runs.append(run)

    for selection_function in selection_functions:
        runs_dict[str(selection_function)].calculate()

    save_to_excel(
        runs_dict,
        encoding,
        file_name if file_name is not None else ff_name,
        fitness_function.__class__.__name__ == "Fconst",
    )

    p_end = time.time()
    print("Program " + file_name + f"({encoding}) calculation (in sec.): " + str((p_end - p_start)))

    return file_name, encoding, runs_dict


def run_algo(selection_functions, fitness_function, file_name, encoding, ff_name, p, i, args):
    results = []

    for selection_function in selection_functions:
        sf_name = str(selection_function)
        sf = selection_function
        optimal = fitness_function.get_optimal(*args, i=i, encoding=encoding)
        folder_name = file_name if file_name is not None else ff_name
        current_run = EvoAlgorithm(
            Population(p.chromosomes.copy(), p.p_m, p.c_m),
            sf,
            fitness_function,
            optimal
        ).run(i, folder_name, encoding, 5)

        results.append((sf_name, current_run))

        if i < 5:
            save_run_plots(folder_name, encoding, sf_name, current_run, i)

    return results
