from math import ceil, floor, log2
from multiprocessing import Pool, cpu_count
from time import time

from numexpr import evaluate, re_evaluate
from scipy.stats.qmc import Sobol


def monte_carlo(f: str, a: int, b: int, iters_log2: int) -> float:
    """ Return sum of the values of function `f` from `a` to `b` over `2^iters` iterations.
    """
    sequence = Sobol(1).random_base2(iters_log2)
    f_sum = 0
    evaluate(f, local_dict={'x': a + (b - a) * 0.5})
    for r in sequence:
        x = a + (b - a) * r[0]
        f_sum += re_evaluate()
    return f_sum


if __name__ == "__main__":
    expr = input("Подынтегральная функция: ")

    integrate_from, integrate_to = map(int, input("Границы интегрирования (a, b): ").split())

    processes = int(input("Количество запускаемых процессов (0 - авто): ")) or cpu_count()
    if processes > 32:
        print("Количество процессов уменьшено до 32")
        processes = 32

    total_iterations = int((integrate_to - integrate_from) * 4096)
    per_process_log2 = floor(log2(total_iterations)) - ceil(log2(processes))
    total_iterations = 2 ** per_process_log2 * processes

    print(f"Вычисление запущено...")
    start = time()
    with Pool(processes) as pool:
        results = pool.starmap(monte_carlo,
                               ((expr, integrate_from, integrate_to, per_process_log2)
                                for i in range(processes)))
    end = time()

    result = (integrate_to - integrate_from) * (sum(results) / total_iterations)

    print(f"\nОпределённый интеграл = {result}")
    print(f"Процессов: {processes}. Общее время вычисления: {(end - start):.3f} с")
