from math import ceil, exp, sin
from multiprocessing import Pool, cpu_count
from random import uniform
from time import time

import scipy.integrate as integrate

TOTAL_ITERATIONS = 50_000_000
INTEGRATE_FROM = 0
INTEGRATE_TO = 28

def f(x: float) -> float:
    """ Function to integrate.
    """
    return exp(x / 2) * sin(x) / (x + 1)**3

def monte_carlo(f, a: int, b: int, iters: int = 1000000) -> float:
    """ Return sum of the values of function `f` from `a` to `b` over `iters` iterations.
    """
    f_sum = 0
    for i in range(iters):
        x = uniform(a, b)
        f_sum += f(x)
    return f_sum

if __name__ == "__main__":
    kernels = cpu_count()
    TOTAL_ITERATIONS = ceil(TOTAL_ITERATIONS / kernels) * kernels
    per_kernel = TOTAL_ITERATIONS // kernels

    start = time()
    with Pool() as pool:
        results = pool.starmap(monte_carlo, ((f, INTEGRATE_FROM, INTEGRATE_TO, per_kernel) for i in range(kernels)))
    end = time()

    total_sum = sum(results)
    average_result = (INTEGRATE_TO - INTEGRATE_FROM) * (total_sum/TOTAL_ITERATIONS)

    print(f"Average definite integral = {average_result}")
    print(f"True definite integral = {integrate.quad(f, INTEGRATE_FROM, INTEGRATE_TO)[0]}")
    print(f"\nCPUs = {kernels}. Total time: {end - start}")
