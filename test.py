import concurrent.futures
from math import ceil
from time import time

import loky
import openturns as ot
import sympy
from sympy.utilities.lambdify import lambdify


TOTAL_ITERATIONS = 10_000_000
INTEGRATE_FROM = 10
INTEGRATE_TO = 15
FUNCTION_STR = "sin(x) / (sin(x)**2 - 4 * sin(x) * cos(x) + 5 * cos(x)**2)"

def monte_carlo(f, a: int, b: int, iters: int) -> float:
    """ Return sum of the values of function `f` from `a` to `b` over `iters` iterations.
    """
    sequence = ot.SobolSequence(1).generate(iters)
    f_sum = 0
    for r in sequence:
        x = a + (b - a) * r[0]
        f_sum += f(x)
    return f_sum


if __name__ == "__main__":
    f = sympy.simplify(FUNCTION_STR)
    f = lambdify(['x'], f)
    print(f"cpu_count = {loky.cpu_count()}\n")

    for processes in [1, 2, 3, 4, 5, 8, 9, 15, 16, 24, 48]:
        TOTAL_ITERATIONS = ceil(TOTAL_ITERATIONS / processes) * processes
        per_process = TOTAL_ITERATIONS // processes

        executor = loky.get_reusable_executor(processes)

        res = 0
        futures = []

        start = time()
        for i in range(processes):
            futures.append(executor.submit(monte_carlo, f=f, a=INTEGRATE_FROM, b=INTEGRATE_TO, iters=per_process))

        for future in concurrent.futures.as_completed(futures):
            res += future.result()
        end = time()

        average_result = (INTEGRATE_TO - INTEGRATE_FROM) * (res / TOTAL_ITERATIONS)

        print(f"Processes: {processes:>2}. Time: {(end - start):.3f} с")
