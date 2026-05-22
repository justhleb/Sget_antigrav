# optimizer.py
"""
Модуль многокритериальной оптимизации распределения парка трамваев по маршрутам.

Использует генетический алгоритм NSGA-II из библиотеки pymoo для нахождения 
оптимального (Парето-эффективного) распределения ограниченного количества трамваев
между маршрутами 20, 48 и 55.

Критерии оптимизации (целевые функции):
1. Минимизация среднего отклонения от целевого интервала (headway MAE) по всем маршрутам.
2. Максимизация общей маржинальной выручки (pymoo минимизирует, поэтому оптимизируется отрицательная выручка).

Ограничения:
1. Суммарное количество выпущенных трамваев не должно превышать бюджет парка N_MAX (67 ТС).
2. Выпуск на каждом конкретном маршруте ограничен контрактным максимумом:
   - Маршрут 20: до 21 ТС
   - Маршрут 48: до 16 ТС
   - Маршрут 55: до 30 ТС
"""

from __future__ import annotations

import logging
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Импорт необходимых классов pymoo для многокритериальной оптимизации
from pymoo.core.problem import ElementwiseProblem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.operators.sampling.rnd import IntegerRandomSampling
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.repair.rounding import RoundingRepair
from pymoo.optimize import minimize
from pymoo.termination import get_termination
from multiprocessing.pool import ThreadPool
import warnings

from simulation.multi_route import MultiRouteSimulation

warnings.filterwarnings("ignore")

# Отключаем лишний вывод логов симуляции во время оптимизации, чтобы не засорять консоль
logging.basicConfig(level=logging.ERROR, force=True)

# ─── Конфигурационные файлы для маршрутных направлений ───────────────────────
ROUTE_PAIRS = {
    "20": ("configs/route_20_fwd_config.json", "configs/route_20_bwd_config.json"),
    "48": ("configs/route_48_fwd_config.json", "configs/route_48_bwd_config.json"),
    "55": ("configs/route_55_fwd_config.json", "configs/route_55_bwd_config.json"),
}

N_ROUTES = len(ROUTE_PAIRS)   # Количество оптимизируемых маршрутов (3)
N_MAX    = 67                  # Общий лимит парка трамваев (21 + 16 + 30)
MAX_PER_ROUTE_MAP = np.array([21, 16, 30])  # Максимально допустимый выпуск по контракту для [20, 48, 55]


# ─── Постановка задачи оптимизации в терминах pymoo ──────────────────────────

class TramFleetProblem(ElementwiseProblem):
    """
    Класс задачи оптимизации распределения парка трамваев.
    
    Переменные решения (x):
        Вектор [n_20, n_48, n_55] из целых чисел.
        Каждая переменная ограничена снизу 0, сверху — контрактным лимитом.
        
    Целевые функции (F):
        pymoo всегда МИНИМИЗИРУЕТ целевые функции. Поэтому:
        F[0] = headway_mae -> минимизируем (среднее отклонение от интервала в минутах)
        F[1] = -total_revenue -> минимизируем отрицательную выручку (что эквивалентно максимизации выручки в рублях)
        
    Ограничения-неравенства (G):
        pymoo требует, чтобы ограничения записывались в виде G(x) <= 0.
        G[0] = sum(x) - N_MAX -> сумма выпущенных трамваев не должна превышать N_MAX.
    """

    def __init__(self, n_max: int = N_MAX, runner=None):
        """
        Инициализирует задачу оптимизации.

        :param n_max: Общий лимит трамваев в парке.
        :param runner: Параллельный обработчик (ThreadPool) для ускорения вычислений.
        """
        super().__init__(
            n_var=N_ROUTES,                     # 3 переменные (по одной на маршрут)
            n_obj=2,                            # 2 целевые функции (интервал и выручка)
            n_ieq_constr=1,                     # 1 ограничение (общий размер парка)
            xl=np.full(N_ROUTES, 0),            # Нижние границы (минимум 0 трамваев)
            xu=MAX_PER_ROUTE_MAP,               # Верхние границы (контрактные максимумы)
            vtype=int,                          # Тип переменных — целые числа
            runner=runner,
        )
        self.n_max = n_max

    def _evaluate(self, x, out, *args, **kwargs):
        """
        Оценивает качество конкретного распределения трамваев с помощью запуска симуляции.

        :param x: Массив/вектор с количеством трамваев: [n_20, n_48, n_55].
        :param out: Словарь для записи вычисленных значений целей (F) и ограничений (G).
        """
        tram_counts = [int(v) for v in x]

        # Инициализируем многомаршрутную симуляцию с заданным распределением парка
        sim = MultiRouteSimulation.from_params(
            ROUTE_PAIRS,
            tram_counts=tram_counts,
            run_dir=None,
        )
        # Запускаем симуляцию в тихом режиме (без построения графиков и сохранения логов)
        sim.run(plot_graphs=False, save_logs=False)

        # Извлекаем агрегированные метрики симуляции
        total_km, headway_mae, total_revenue, _ = sim.get_objectives()

        # Формируем вектор целевых функций (выручку берем с минусом для максимизации)
        out["F"] = np.array([headway_mae, -total_revenue], dtype=float)
        # Формируем ограничение на вместимость парка (сумма ТС - лимит <= 0)
        out["G"] = np.array([sum(tram_counts) - self.n_max], dtype=float)

        # Явно освобождаем память от объекта симуляции
        del sim


# ─── Запуск алгоритма NSGA-II ────────────────────────────────────────────────

def run_nsga2(
    n_max:    int = N_MAX,
    pop_size: int = 25,
    n_gen:    int = 5,
    seed:     int = 42,
    out_dir:  str = "outputs/nsga2",
) -> tuple:
    """
    Запускает многокритериальный генетический алгоритм NSGA-II.

    :param n_max: Общий предел количества трамваев.
    :param pop_size: Размер популяции в одном поколении.
    :param n_gen: Количество поколений эволюции.
    :param seed: Начальное значение генератора случайных чисел для воспроизводимости.
    :param out_dir: Базовая директория для сохранения результатов.
    :return: Объект с результатами оптимизации от pymoo.
    """
    from datetime import datetime
    # Формируем имя уникальной папки запуска на основе текущего времени
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = os.path.join(out_dir, f"run_{ts}")
    os.makedirs(run_dir, exist_ok=True)

    # Настраиваем параллельные вычисления через пул потоков
    n_threads = 8
    pool = ThreadPool(n_threads)
    runner = pool.starmap

    # Создаем объект задачи оптимизации
    problem = TramFleetProblem(n_max=n_max, runner=runner)

    # Инициализируем алгоритм NSGA-II с операторами для целочисленного кодирования
    algorithm = NSGA2(
        pop_size=pop_size,                      # Число особей (вариантов распределения) в популяции
        sampling=IntegerRandomSampling(),      # Целочисленная случайная начальная популяция
        crossover=SBX(prob=0.9, eta=15, vtype=float, repair=RoundingRepair()), # Скрещивание с округлением
        mutation=PM(eta=20, vtype=float, repair=RoundingRepair()),             # Мутация с округлением
        eliminate_duplicates=True,             # Исключение одинаковых решений в популяции
    )

    # Критерий остановки — достижение заданного числа поколений
    termination = get_termination("n_gen", n_gen)

    print(f"Запуск NSGA-II: pop={pop_size}, gen={n_gen}, n_max={n_max}")
    print(f"Всего evaluations: ~{pop_size * n_gen}")

    # Запуск оптимизационного процесса
    res = minimize(
        problem,
        algorithm,
        termination,
        seed=seed,
        verbose=True,
        save_history=False,
    )

    # Обязательно закрываем и очищаем пул потоков после окончания оптимизации
    pool.close()
    pool.join()

    # Сохраняем Парето-фронт в CSV и визуализируем результаты
    _save_results(res, run_dir)

    return res


def _save_results(res, out_dir: str):
    """
    Сохраняет найденные решения Парето-фронта в файл CSV и строит график.

    :param res: Объект результатов от pymoo.
    :param out_dir: Папка для записи файлов.
    """
    X = res.X   # Векторы распределения трамваев: [n_20, n_48, n_55]
    F = res.F   # Векторы целей: [headway_mae_min, -total_revenue]

    if X is None or F is None:
        print("\n[Внимание] Не найдено ни одного допустимого решения.")
        return

    # Формируем DataFrame с результатами
    df = pd.DataFrame(
        np.hstack([X, F]),
        columns=["n_20", "n_48", "n_55", "headway_mae_min", "total_revenue_neg"],
    )
    # Переводим выручку обратно в положительные числа
    df["total_revenue"] = -df["total_revenue_neg"]
    df = df.drop(columns=["total_revenue_neg"])
    # Добавляем суммарное количество трамваев для проверки ограничения
    df["total_trams"] = df[["n_20", "n_48", "n_55"]].sum(axis=1)

    # Сортируем решения по убыванию выручки (для удобства чтения)
    df = df.sort_values("total_revenue", ascending=False)
    csv_path = os.path.join(out_dir, "pareto_front.csv")
    df.to_csv(csv_path, index=False)
    
    print(f"\nPareto-фронт сохранён: {csv_path}")
    print(df.to_string(index=False))

    # Пытаемся построить графики Парето-фронта
    plots_dir = os.path.join(out_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    try:
        from plot_pareto import plot_pareto
        plot_pareto(csv_path=csv_path, out_dir=plots_dir)
    except Exception as e:
        print(f"Ошибка при построении графиков Парето: {e}")


# ─── Точка входа для локального тестирования оптимизатора ───────────────────

if __name__ == "__main__":
    run_nsga2(n_max=N_MAX, pop_size=25, n_gen=5)
