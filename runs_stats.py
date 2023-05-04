from statistics import mean
import math


def sigma(items):
    if len(items) < 2:
        return None
    avg_x = mean(items)
    e = sum([math.pow(item - avg_x, 2) for item in items])
    return math.sqrt(e / (len(items) - 1))


class RunsStats:
    def __init__(self):
        self.runs = []
        self.success_percentage = 0
        self.min_NI = None
        self.max_NI = None
        self.avg_NI = None
        self.sigma_NI = None
        self.min_I_min = None
        self.NI_I_min = None
        self.max_I_max = None
        self.NI_I_max = None
        self.avg_I_min = None
        self.avg_I_max = None
        self.avg_I_avg = None
        self.sigma_I_min = None
        self.sigma_I_max = None
        self.sigma_I_avg = None
        self.avg_gr_early = None
        self.min_gr_early = None
        self.max_gr_early = None
        self.avg_gr_late = None
        self.min_gr_late = None
        self.max_gr_late = None
        self.avg_gr_avg = None
        self.min_gr_avg = None
        self.max_gr_avg = None
        self.min_rr_min = None
        self.NI_rr_min = None
        self.max_rr_max = None
        self.NI_rr_max = None
        self.avg_rr_min = None
        self.avg_rr_max = None
        self.avg_rr_avg = None
        self.sigma_rr_min = None
        self.sigma_rr_max = None
        self.sigma_rr_avg = None
        self.min_teta_min = None
        self.NI_teta_min = None
        self.max_teta_max = None
        self.NI_teta_max = None
        self.avg_teta_min = None
        self.avg_teta_max = None
        self.avg_teta_avg = None
        self.sigma_teta_min = None
        self.sigma_teta_max = None
        self.sigma_teta_avg = None
        self.min_s_min = None
        self.NI_s_min = None
        self.max_s_max = None
        self.NI_s_max = None
        self.avg_s_min = None
        self.avg_s_max = None
        self.avg_s_avg = None
        self.noise_suc = None
        self.noise_num0 = None
        self.noise_num1 = None
        self.noise_NI_min = None
        self.noise_NI_max = None
        self.noise_NI_avg = None

    def calculate(self):
        successful_runs = [run for run in self.runs if run.is_successful]
        self.success_percentage = (len(successful_runs) / len(self.runs)) * 100
        self.calculate_convergence_stats(successful_runs)
        self.calculate_i_stats(successful_runs)
        self.calculate_gr_stats(successful_runs)
        self.calculate_rr_stats(successful_runs)
        self.calculate_teta_stats(successful_runs)
        self.calculate_s_stats(successful_runs)

        if all((run.noise_stats is not None for run in self.runs)):
            self.calculate_noise_stats()

    def calculate_noise_stats(self):
        suc_noise_stats = [
            run.noise_stats for run in self.runs if run.noise_stats.conv_to is not None
        ]
        self.noise_suc = len(suc_noise_stats) / len(self.runs)
        if self.noise_suc != 0 and len(suc_noise_stats) > 0:
            self.noise_num0 = len(
                [ns for ns in suc_noise_stats if ns.conv_to == 0]
            ) / len(suc_noise_stats)
            self.noise_num1 = 1 - self.noise_num0
        nis = [ns.NI for ns in suc_noise_stats if ns.NI is not None]
        if len(nis) > 0:
            self.noise_NI_min = min(nis)
            self.noise_NI_max = max(nis)
            self.noise_NI_avg = mean(nis)

    def calculate_convergence_stats(self, successful_runs):
        convergence_iterations = [run.pressure_stats.NI for run in successful_runs]
        if len(convergence_iterations) > 0:
            self.min_NI = min(convergence_iterations)
            self.max_NI = max(convergence_iterations)
            self.avg_NI = mean(convergence_iterations)
            self.sigma_NI = sigma(convergence_iterations)

    def calculate_i_stats(self, successful_runs):
        i_min_list = [run.pressure_stats.i_min for run in successful_runs]
        if len(i_min_list) > 0:
            self.min_I_min = min(i_min_list)
            self.sigma_I_min = sigma(i_min_list)
            self.avg_I_min = mean(i_min_list)
        i_max_list = [run.pressure_stats.i_max for run in successful_runs]
        if len(i_max_list) > 0:
            self.max_I_max = max(i_max_list)
            self.avg_I_max = mean(i_max_list)
            self.sigma_I_max = sigma(i_max_list)
        i_avg_list = [run.pressure_stats.i_avg for run in successful_runs]
        if len(i_avg_list) > 0:
            self.avg_I_avg = mean(i_avg_list)
            self.sigma_I_avg = sigma(i_avg_list)
        ni_i_min_list = [run.pressure_stats.i_imin for run in successful_runs]
        if len(ni_i_min_list):
            self.NI_I_min = min(ni_i_min_list)
        ni_i_max_list = [run.pressure_stats.i_imax for run in successful_runs]
        if len(ni_i_max_list) > 0:
            self.NI_I_max = max(ni_i_max_list)

    def calculate_gr_stats(self, successful_runs):
        gre_list = [
            run.pressure_stats.gre
            for run in successful_runs
            if run.pressure_stats.gre is not None
        ]
        grl_list = [
            run.pressure_stats.grl
            for run in successful_runs
            if run.pressure_stats.grl is not None
        ]
        gra_list = [
            run.pressure_stats.gra
            for run in successful_runs
            if run.pressure_stats.gra is not None
        ]
        if len(gre_list) > 0:
            self.avg_gr_early = mean(gre_list)
            self.min_gr_early = min(gre_list)
            self.max_gr_early = max(gre_list)
        if len(grl_list) > 0:
            self.avg_gr_late = mean(grl_list)
            self.min_gr_late = min(grl_list)
            self.max_gr_late = max(grl_list)
        if len(gra_list):
            self.avg_gr_avg = mean(gra_list)
            self.min_gr_avg = min(gra_list)
            self.max_gr_avg = max(gra_list)

    def calculate_rr_stats(self, successful_runs):
        rr_min_list = [run.reproduction_stats.rr_min for run in successful_runs]
        if len(rr_min_list) > 0:
            self.min_rr_min = min(rr_min_list)
            self.avg_rr_min = mean(rr_min_list)
            self.sigma_rr_min = sigma(rr_min_list)
        rr_max_list = [run.reproduction_stats.rr_max for run in successful_runs]
        if len(rr_max_list) > 0:
            self.max_rr_max = max(rr_max_list)
            self.avg_rr_max = mean(rr_max_list)
            self.sigma_rr_max = sigma(rr_max_list)
        rr_avg_list = [run.reproduction_stats.rr_avg for run in successful_runs]
        if len(rr_avg_list) > 0:
            self.avg_rr_avg = mean(rr_avg_list)
            self.sigma_rr_avg = sigma(rr_avg_list)
        ni_rr_min_list = [run.reproduction_stats.ni_rr_min for run in successful_runs]
        if len(ni_rr_min_list) > 0:
            self.NI_rr_min = min(ni_rr_min_list)
        ni_rr_max_list = [run.reproduction_stats.ni_rr_max for run in successful_runs]
        if len(ni_rr_max_list) > 0:
            self.NI_rr_max = max(ni_rr_max_list)

    def calculate_teta_stats(self, successful_runs):
        teta_min_list = [run.reproduction_stats.teta_min for run in successful_runs]
        if len(teta_min_list) > 0:
            self.min_teta_min = min(teta_min_list)
            self.avg_teta_min = mean(teta_min_list)
            self.sigma_teta_min = sigma(teta_min_list)
        teta_max_list = [run.reproduction_stats.teta_max for run in successful_runs]
        if len(teta_max_list) > 0:
            self.max_teta_max = max(teta_max_list)
            self.avg_teta_max = mean(teta_max_list)
            self.sigma_teta_max = sigma(teta_max_list)
        teta_avg_list = [run.reproduction_stats.teta_avg for run in successful_runs]
        if len(teta_avg_list) > 0:
            self.avg_teta_avg = mean(teta_avg_list)
            self.sigma_teta_avg = sigma(teta_avg_list)
        ni_teta_min_list = [
            run.reproduction_stats.ni_teta_min for run in successful_runs
        ]
        if len(ni_teta_min_list) > 0:
            self.NI_teta_min = min(ni_teta_min_list)
        ni_teta_max_list = [
            run.reproduction_stats.ni_teta_max for run in successful_runs
        ]
        if len(ni_teta_max_list) > 0:
            self.NI_teta_max = max(ni_teta_max_list)

    def calculate_s_stats(self, successful_runs):
        s_min_list = [run.selection_diff_stats.s_min for run in successful_runs]
        if len(s_min_list) > 0:
            self.min_s_min = min(s_min_list)
            self.avg_s_min = mean(s_min_list)
        s_max_list = [run.selection_diff_stats.s_max for run in successful_runs]
        if len(s_max_list) > 0:
            self.max_s_max = max(s_max_list)
            self.avg_s_max = mean(s_max_list)
        s_avg_list = [run.selection_diff_stats.s_avg for run in successful_runs]
        if len(s_avg_list) > 0:
            self.avg_s_avg = mean(s_avg_list)
        ni_s_min_list = [run.selection_diff_stats.ni_s_min for run in successful_runs]
        if len(ni_s_min_list) > 0:
            self.NI_s_min = min(ni_s_min_list)
        ni_s_max_list = [run.selection_diff_stats.ni_s_max for run in successful_runs]
        if len(ni_s_max_list) > 0:
            self.NI_s_max = max(ni_s_max_list)

    def as_dict(self):
        return {
            "Avg_NI": [self.avg_NI],
            "Avg_I_min": [self.avg_I_min],
            "Avg_I_max": [self.avg_I_max],
            "Avg_I_avg": [self.avg_I_avg],
            "AvgGR_early": [self.avg_gr_early],
            "AvgGR_late": [self.avg_gr_late],
            "AvgGR_avg": [self.avg_gr_avg],
            "Avg_RR_min": [self.avg_rr_min],
            "Avg_RR_max": [self.avg_rr_max],
            "Avg_RR_avg": [self.avg_rr_avg],
            "Avg_Teta_min": [self.avg_teta_min],
            "Avg_Teta_max": [self.avg_teta_max],
            "Avg_Teta_avg": [self.avg_teta_avg],
            "Avg_s_min": [self.avg_s_min],
            "Avg_s_max": [self.avg_s_max],
            "Avg_s_avg": [self.avg_s_avg],
            "Max_NI": [self.max_NI],
            "NI_I_max": [self.NI_I_max],
            "Max_I_max": [self.max_I_max],
            "MaxGR_early": [self.max_gr_early],
            "MaxGR_late": [self.max_gr_late],
            "MaxGR_avg": [self.max_gr_avg],
            "NI_RR_max": [self.NI_rr_max],
            "Max_RR_max": [self.max_rr_max],
            "NI_Teta_max": [self.NI_teta_max],
            "Max_Teta_max": [self.max_teta_max],
            "NI_s_max": [self.NI_s_max],
            "Max_s_max": [self.max_s_max],
            "Min_NI": [self.min_NI],
            "NI_I_min": [self.NI_I_min],
            "Min_I_min": [self.min_I_min],
            "MinGR_early": [self.min_gr_early],
            "MinGR_late": [self.min_gr_late],
            "MinGR_avg": [self.min_gr_avg],
            "NI_RR_min": [self.NI_rr_min],
            "Min_RR_min": [self.min_rr_min],
            "NI_Teta_min": [self.NI_teta_min],
            "Min_Teta_min": [self.min_teta_min],
            "NI_s_min": [self.NI_s_min],
            "Min_s_min": [self.min_s_min],
            "Sigma_NI": [self.sigma_NI],
            "Sigma_I_min": [self.sigma_I_min],
            "Sigma_I_max": [self.sigma_I_max],
            "Sigma_I_avg": [self.sigma_I_avg],
            "Sigma_RR_min": [self.sigma_rr_min],
            "Sigma_RR_max": [self.sigma_rr_max],
            "Sigma_RR_avg": [self.sigma_rr_avg],
            "Sigma_Teta_min": [self.sigma_teta_min],
            "Sigma_Teta_max": [self.sigma_teta_max],
            "Sigma_Teta_avg": [self.sigma_teta_avg],
            "Suc": [self.success_percentage],
        }

    def as_noise_dict(self):
        return {
            "Noise Suc": [self.noise_suc],
            "Noise 0": [self.noise_num0],
            "Moise 1": [self.noise_num1],
            "Noise NI min": [self.noise_NI_min],
            "Noise NI max": [self.noise_NI_max],
            "Noise NI avg": [self.noise_NI_avg],
        }

    def __str__(self):
        return (
            "Suc: "
            + str(self.success_percentage)
            + "%"
            + "\nMin: "
            + str(self.min_NI)
            + "\nMax: "
            + str(self.max_NI)
            + "\nAvg: "
            + str(self.avg_NI)
        )
