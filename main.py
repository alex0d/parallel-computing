import concurrent.futures
from math import ceil, exp, sin
from random import uniform
from time import time

import scipy.integrate as integrate
import sympy
from sympy.utilities.lambdify import lambdify, implemented_function
import loky

TOTAL_ITERATIONS = 50_000_000
INTEGRATE_FROM = 0
INTEGRATE_TO = 28


def monte_carlo(f, a: int, b: int, iters: int = 1000000) -> float:
    """ Return sum of the values of function `f` from `a` to `b` over `iters` iterations.
    """
    f_sum = 0
    for i in range(iters):
        x = uniform(a, b)
        f_sum += f(x)
    return f_sum


if __name__ == "__main__":
    f = input("Подынтегральная функция: ")
    f = sympy.simplify(f)

    kernels = int(input("Введите количество запускаемых потоков (0 - авто): ")) or loky.cpu_count()
    TOTAL_ITERATIONS = ceil(TOTAL_ITERATIONS / kernels) * kernels
    per_kernel = TOTAL_ITERATIONS // kernels

    f_lambda = lambdify(['x'], f)

    executor = loky.get_reusable_executor(max_workers=kernels)
    start = time()
    # executor.map(monte_carlo, [(f_lambda, INTEGRATE_FROM, INTEGRATE_TO, per_kernel) for i in range(kernels)])
    # executor.shutdown(wait=True)
    res = 0
    futures = []
    for i in range(8):
        futures.append(executor.submit(monte_carlo, f=f_lambda, a=INTEGRATE_FROM, b=INTEGRATE_TO, iters=per_kernel))
    for future in concurrent.futures.as_completed(futures):
        res += future.result()

    end = time()

    # total_sum = sum(results)
    average_result = (INTEGRATE_TO - INTEGRATE_FROM) * (res / TOTAL_ITERATIONS)

    print(f"Average definite integral = {average_result}")
    # print(f"True definite integral = {integrate.quad(f, INTEGRATE_FROM, INTEGRATE_TO)[0]}")
    print(f"\nCPUs = {kernels}. Total time: {end - start}")
