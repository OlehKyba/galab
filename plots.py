import matplotlib.pyplot as plt
from constants import N
import os


def save_line_plot(fitness_func_name, encoding, func_name, data, file_name, y_label, iteration):
    path = f"{fitness_func_name}/{encoding}" if encoding else fitness_func_name
    path += f"/{N}/{func_name}/{iteration}"
    if not os.path.exists(path):
        os.makedirs(path)

    x = list(range(1, len(data) + 1))
    plt.plot(x, data, label=func_name)
    plt.ylabel(y_label)
    plt.xlabel("generation")
    plt.ylim(ymin=0)
    plt.xlim(xmin=1)
    plt.legend()
    plt.savefig(path + "/" + file_name + ".png")
    plt.close()


def save_lines_plot(
    fitness_func_name, encoding, func_name, data_arr, label_arr, file_name, y_label, iteration
):
    path = f"{fitness_func_name}/{encoding}" if encoding else fitness_func_name
    path += f"/{N}/{func_name}/{iteration}"

    if not os.path.exists(path):
        os.makedirs(path)

    for i in range(0, len(data_arr)):
        data = data_arr[i]
        label = label_arr[i]
        x = list(range(1, len(data) + 1))
        plt.plot(x, data, label=label)

    plt.ylabel(y_label)
    plt.xlabel("generation")
    plt.ylim(ymin=0)
    plt.xlim(xmin=1)
    plt.legend()
    plt.savefig(path + "/" + file_name + ".png")
    plt.close()
