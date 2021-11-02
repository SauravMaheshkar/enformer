import numpy as np

__all__ = ["exponential_linspace_int"]


def exponential_linspace_int(start, end, num, divisible_by=1):
    def _round(x):
        return int(np.round(x / divisible_by) * divisible_by)

    base = np.exp(np.log(end / start) / (num - 1))
    return [_round(start * base ** i) for i in range(num)]
