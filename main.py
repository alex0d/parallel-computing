import concurrent.futures
from time import time

import loky
import openturns as ot
import sympy
from sympy.utilities.lambdify import lambdify


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
    total_iterations = int((integrate_to - integrate_from) * 10_000)
    per_process = total_iterations // processes

    total_sum = 0
    futures = []

    print(f"Вычисление запущено...")
    start = time()
    with loky.get_reusable_executor(processes) as executor:
        for i in range(processes):
            futures.append(executor.submit(monte_carlo, f=f, a=integrate_from, b=integrate_to, iters=per_process))

        for future in concurrent.futures.as_completed(futures):
            total_sum += future.result()
    end = time()

    average_result = (integrate_to - integrate_from) * (total_sum / total_iterations)

    print(f"\nОпределённый интеграл = {average_result}")
    print(f"Процессов: {processes}. Общее время вычисления: {(end - start):.3f} с")