import time
from multiprocessing import Pool, cpu_count
from constants import P_M, P_C, P_M_DICT, env
from functions import *
from selection.rws import RankExponentialRWS
from selection.sus import RankExponentialSUS
from plots import *
from program import main
from excel import save_avg_to_excel
from population import generate_population_seeds

import matplotlib

matplotlib.rcParams['axes.formatter.useoffset'] = False


release_sm = [RankExponentialSUS, RankExponentialRWS]
testing_sm = [RankExponentialSUS, RankExponentialRWS]

C1 = 0.809
C2 = 0.9
C3 = 0.955
C4 = 0.979

fconst_seeds = generate_population_seeds()
fhd_seeds = generate_population_seeds()
fx2_seeds = generate_population_seeds()
f5122subx2_seeds = generate_population_seeds()

# selection_methods = testing_sm if env == "test" else release_sm
selection_methods = [
    RankExponentialRWS(c=C1),
    RankExponentialRWS(c=C2),
    RankExponentialRWS(c=C3),
    RankExponentialRWS(c=C4),

    RankExponentialSUS(c=C1),
    RankExponentialSUS(c=C2),
    RankExponentialSUS(c=C3),
    RankExponentialSUS(c=C4),
]
# release_functions = [
#     (FHD(100), selection_methods, "FHD", N, 100, 0, 0),
#     (FHD(100), selection_methods, "FHD_pm", N, 100, P_M, 0),
#     (FHD(100), selection_methods, "FHD_pc", N, 100, 0, P_C),
#     (FHD(100), selection_methods, "FHD_pmpc", N, 100, P_M, P_C),
#     (Fx2(0, 10.23), selection_methods, "Fx2", N, 10, 0, 0),
#     (Fx2(0, 10.23), selection_methods, "Fx2_pm", N, 10, P_M, 0),
#     (Fx2(0, 10.23), selection_methods, "Fx2_pc", N, 10, 0, P_C),
#     (Fx2(0, 10.23), selection_methods, "Fx2_pmpc", N, 10, P_M, P_C),
#     (F5122subx2(-5.11, 5.12), selection_methods, "512subx2", N, 10, 0, 0),
#     (F5122subx2(-5.11, 5.12), selection_methods, "512subx2_pm", N, 10, P_M, 0),
#     (F5122subx2(-5.11, 5.12), selection_methods, "512subx2_pc", N, 10, 0, P_C),
#     (F5122subx2(-5.11, 5.12), selection_methods, "512subx2_pmpc", N, 10, P_M, P_C),
#     (Fecx(0, 10.23, 0.25), selection_methods, "Fec025x", N, 10, 0, 0),
#     (Fecx(0, 10.23, 0.25), selection_methods, "Fec025x_pm", N, 10, P_M, 0),
#     (Fecx(0, 10.23, 0.25), selection_methods, "Fec025x_pc", N, 10, 0, P_C),
#     (Fecx(0, 10.23, 0.25), selection_methods, "Fec025x_pmpc", N, 10, P_M, P_C),
#     (Fecx(0, 10.23, 1), selection_methods, "Fec1x", N, 10, 0, 0),
#     (Fecx(0, 10.23, 1), selection_methods, "Fec1x_pm", N, 10, P_M, 0),
#     (Fecx(0, 10.23, 1), selection_methods, "Fec1x_pc", N, 10, 0, P_C),
#     (Fecx(0, 10.23, 1), selection_methods, "Fec1x_pmpc", N, 10, P_M, P_C),
#     (Fecx(0, 10.23, 2), selection_methods, "Fec2x", N, 10, 0, 0),
#     (Fecx(0, 10.23, 2), selection_methods, "Fec2x_pm", N, 10, P_M, 0),
#     (Fecx(0, 10.23, 2), selection_methods, "Fec2x_pc", N, 10, 0, P_C),
#     (Fecx(0, 10.23, 2), selection_methods, "Fec2x_pmpc", N, 10, P_M, P_C),
# ]
release_functions = (
    (Fconst(fconst_seeds), selection_methods, "Fconst", None, N, 100, 0, 0),
    (Fconst(fconst_seeds), selection_methods, "Fconst_pc", None, N, 100, 0, P_C),
    (Fconst(fconst_seeds), selection_methods, "Fconst_pm", None, N, 100, 0.00001, 0),
    (Fconst(fconst_seeds), selection_methods, "Fconst_pmpc", None, N, 100, 0.00001, P_C),
    (FHD(100, fhd_seeds), selection_methods, "FHD", None, N, 100, 0, 0),
    (FHD(100, fhd_seeds), selection_methods, "FHD_pc", None, N, 100, 0, P_C),
    (FHD(100, fhd_seeds), selection_methods, "FHD_pm", None, N, 100, 0.00001, 0),
    (FHD(100, fhd_seeds), selection_methods, "FHD_pmpc", None, N, 100, 0.00001, P_C),
    (Fx2(0, 10.23, fx2_seeds), selection_methods, "Fx2", "binary", N, 10, 0, 0),
    (Fx2(0, 10.23, fx2_seeds), selection_methods, "Fx2_pm", "binary", N, 10, 0.0001, 0),
    (Fx2(0, 10.23, fx2_seeds), selection_methods, "Fx2_pc", "binary", N, 10, 0, P_C),
    (Fx2(0, 10.23, fx2_seeds), selection_methods, "Fx2_pmpc", "binary", N, 10, 0.0001, P_C),
    (Fx2(0, 10.23, fx2_seeds), selection_methods, "Fx2", "gray", N, 10, 0, 0),
    (Fx2(0, 10.23, fx2_seeds), selection_methods, "Fx2_pm", "gray", N, 10, 0.0001, 0),
    (Fx2(0, 10.23, fx2_seeds), selection_methods, "Fx2_pc", "gray", N, 10, 0, P_C),
    (Fx2(0, 10.23, fx2_seeds), selection_methods, "Fx2_pmpc", "gray", N, 10, 0.0001, P_C),
    (F5122subx2(-5.11, 5.12, f5122subx2_seeds), selection_methods, "512subx2", "binary", N, 10, 0, 0),
    (F5122subx2(-5.11, 5.12, f5122subx2_seeds), selection_methods, "512subx2_pm", "binary", N, 10, 0.0001, 0),
    (F5122subx2(-5.11, 5.12, f5122subx2_seeds), selection_methods, "512subx2_pc", "binary", N, 10, 0, P_C),
    (F5122subx2(-5.11, 5.12, f5122subx2_seeds), selection_methods, "512subx2_pmpc", "binary", N, 10, 0.0001, P_C),
    (F5122subx2(-5.11, 5.12, f5122subx2_seeds), selection_methods, "512subx2", "gray", N, 10, 0, 0),
    (F5122subx2(-5.11, 5.12, f5122subx2_seeds), selection_methods, "512subx2_pm", "gray", N, 10, 0.0001, 0),
    (F5122subx2(-5.11, 5.12, f5122subx2_seeds), selection_methods, "512subx2_pc", "gray", N, 10, 0, P_C),
    (F5122subx2(-5.11, 5.12, f5122subx2_seeds), selection_methods, "512subx2_pmpc", "gray", N, 10, 0.0001, P_C),
)


test_functions = [
    (FHD(100, population_seeds=fhd_seeds), selection_methods, "FHD", None, N, 100, 0, 0),
    # (Fx2(0, 10.23, population_seeds=fx2_seeds), selection_methods, "Fx2_pmpc", "binary", N, 10, 0.0001, P_C),
    # (Fx2(0, 10.23, population_seeds=fx2_seeds), selection_methods, "Fx2_pmpc", "gray", N, 10, 0.0001, P_C)
    # (Fconst(), selection_methods, "Fconst", N, 100, 0, 0),
    # (FHD(100), selection_methods, "FHD_pmpc", N, 100, 0.00001, P_C),
    # (Fx2(0, 10.23), selection_methods, "Fx2_pmpc", N, 10, 0.0001, P_C),
    # (F5122subx2(-5.11, 5.12), selection_methods, "512subx2_pmpc", N, 10,  0.0001, P_C),
]
functions = test_functions if env == "test" else release_functions


if __name__ == "__main__":
    p_start = time.time()

    func_runs = {}
    noise_runs = {}

    print("Program calculation start ...")

    with Pool(cpu_count()) as pool:
        for config in functions:
            fitness_function, selection_functions, file_name, encoding, *args = config
            _, _, runs_dict = main(fitness_function, selection_functions, file_name, encoding, pool, *args)
            func_runs[(file_name, encoding)] = runs_dict

        # for file_name, encoding, run in p.starmap(main, functions):
        #     func_runs[(file_name, encoding)] = run

        save_avg_to_excel(func_runs, noise_runs)

    p_end = time.time()
    print("Program calculation (in sec.): " + str((p_end - p_start)))
