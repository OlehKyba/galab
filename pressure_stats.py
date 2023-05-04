from statistics import mean


def s(fs, f):
    return fs - f


class PressureStats:
    def __init__(self):
        self.num_of_best = []
        self.f_best = []
        self.grs = []
        self.intensities = []
        self.NI = None
        self.f_found = None
        self.f_avg = None
        self.grl = None
        self.gre = None
        self.gra = None
        self.grli = None
        self.i_min = None
        self.i_max = None
        self.i_avg = None
        self.i_imin = None
        self.i_imax = None

    def __str__(self):
        return (
            "\nNI: "
            + str(self.NI)
            + "\nF found: "
            + str(self.f_found)
            + "\nF avg: "
            + str(self.f_avg)
            + "\nI min: "
            + str(self.i_min)
            + "\nNI I min: "
            + str(self.i_imin)
            + "\nI max: "
            + str(self.i_max)
            + "\nNI I max: "
            + str(self.i_imax)
            + "\nGR early: "
            + str(self.gre)
            + "\nGR avg: "
            + str(self.gra)
            + "\nGR late: "
            + str(self.grl)
            + "\nNI GR late: "
            + str(self.grli)
        )

    def as_dict(self):
        return {
            "NI": [self.NI],
            "F_found": [self.f_found],
            "F_avg": [self.f_avg],
            "I_min": [self.i_min],
            "NI_I_min": [self.i_imin],
            "I_max": [self.i_max],
            "NI_I_max": [self.i_imax],
            "I_avg": [self.i_avg],
            "GR_early": [self.gre],
            "GR_avg": [self.gra],
            "GR_late": [self.grl],
            "NI_GR_late": [self.grli],
        }

    def calculate_intensity_coefficients(self):
        nni = [x for x in self.intensities if x is not None]
        if len(nni) > 0:
            self.i_min = min(nni)
            self.i_max = max(nni)
            self.i_avg = mean(nni)
            self.i_imin = self.intensities.index(self.i_min)
            self.i_imax = self.intensities.index(self.i_max)

    def calculate_growth_rate_coefficients(self):
        if len(self.grs) > 0:
            self.gre = self.grs[1]
            self.gra = mean(self.grs)

    def calculate(self):
        self.calculate_intensity_coefficients()
        self.calculate_growth_rate_coefficients()

    @staticmethod
    def calculate_intensity(fs, f, std):
        if std == 0:
            return 1
        if fs is None or f is None or std is None:
            return None
        return s(fs, f) / std

    @staticmethod
    def calculate_growth_rate(current, previous, f_best_current, f_best_previous):
        if f_best_current == f_best_previous and previous > 0:
            return current / previous
        else:
            return 0
