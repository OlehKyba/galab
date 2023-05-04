from statistics import mean


class SelectionDiffStats:
    def __init__(self):
        self.s_list = []
        self.s_min = None
        self.ni_s_min = None
        self.s_max = None
        self.ni_s_max = None
        self.s_avg = None

    def calculate(self):
        self.s_min = min(self.s_list)
        self.ni_s_min = self.s_list.index(self.s_min)
        self.s_max = max(self.s_list)
        self.ni_s_max = self.s_list.index(self.s_max)
        self.s_avg = mean(self.s_list)

    def __str__(self):
        return (
            "\ns_min: "
            + str(self.s_min)
            + " NI_s_min: "
            + str(self.ni_s_min)
            + "\ns_max: "
            + str(self.s_max)
            + " NI_s_max: "
            + str(self.ni_s_max)
            + "\ns_avg: "
            + str(self.s_avg)
        )

    def as_dict(self):
        return {
            "s_min": [self.s_min],
            "NI_s_min": [self.ni_s_min],
            "s_max": [self.s_max],
            "NI_s_max": [self.ni_s_max],
            "s_avg": [self.s_avg],
        }
