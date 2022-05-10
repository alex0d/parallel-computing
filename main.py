import concurrent.futures
from math import ceil
from time import time

import loky
import openturns as ot
import sympy
from sympy.utilities.lambdify import lambdify


TOTAL_ITERATIONS = 10_000_000


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
    f = input("Подынтегральная функция: ")
    f = sympy.simplify(f)
    f = lambdify(['x'], f)

    integrate_from, integrate_to = map(int, input("Границы интегрирования (a, b): ").split())

    processes = int(input("Количество запускаемых процессов (0 - авто): ")) or loky.cpu_count()
    TOTAL_ITERATIONS = ceil(TOTAL_ITERATIONS / processes) * processes
    per_process = TOTAL_ITERATIONS // processes

    executor = loky.get_reusable_executor(processes)

    res = 0
    futures = []

    print(f"Вычисление запущено...")
    start = time()
    for i in range(processes):
        futures.append(executor.submit(monte_carlo, f=f, a=integrate_from, b=integrate_to, iters=per_process))

    for future in concurrent.futures.as_completed(futures):
        res += future.result()
    end = time()

    average_result = (integrate_to - integrate_from) * (res / TOTAL_ITERATIONS)

    print(f"\nОпределённый интеграл = {average_result}")
    print(f"Процессов: {processes}. Общее время вычисления: {(end - start):.3f} с")
