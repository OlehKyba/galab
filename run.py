class Run:
    def __init__(
        self,
        avg_fitness_list=None,
        std_fitness_list=None,
        pressure_stats=None,
        reproduction_stats=None,
        selection_diff_stats=None,
        noise_stats=None,
        is_successful=None,
    ):
        self.avg_fitness_list = avg_fitness_list
        self.std_fitness_list = std_fitness_list
        self.pressure_stats = pressure_stats
        self.reproduction_stats = reproduction_stats
        self.selection_diff_stats = selection_diff_stats
        self.noise_stats = noise_stats
        self.is_successful = is_successful
