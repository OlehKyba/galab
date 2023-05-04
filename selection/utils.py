from chromosome import Chromosome


def create_wheel(
    probabilities_with_chromosomes: list[tuple[float, Chromosome]]
) -> list[tuple[tuple[float, float], Chromosome]]:
    wheel: list[tuple[tuple[float, float], Chromosome]] = []
    segment = 0

    for probability, chromosome in probabilities_with_chromosomes:
        segment_from, segment_to = segment, segment + probability
        wheel.append(((segment_from, segment_to), chromosome))
        segment += probability

    if wheel[-1][0][1] != 1:
        (start, _), chromosome = wheel[-1]
        wheel[-1] = ((start, 1), chromosome)

    return wheel


def binary_search_by_wheel(
    wheel: list[tuple[tuple[float, float], Chromosome]],
    wheel_point: float,
    left: int = 0,
) -> tuple[int, Chromosome]:
    right = len(wheel) - 1
    while left <= right:
        mid = (left + right) // 2
        probability_range, chromosome = wheel[mid]
        if wheel_point < probability_range[0]:
            right = mid - 1
        elif wheel_point > probability_range[1]:
            left = mid + 1
        else:
            return mid, chromosome

    raise ValueError(f"Could not find {wheel_point=} in {wheel=}")
