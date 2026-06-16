# optimizer.py
"""
Модуль многокритериальной оптимизации распределения парка трамваев по маршрутам.

Использует генетический алгоритм NSGA-II из библиотеки pymoo для нахождения 
оптимального (Парето-эффективного) распределения ограниченного количества трамваев
между маршрутами 20, 48 и 55.

Критерии оптимизации (целевые функции):
1. Минимизация среднего отклонения от целевого интервала (headway MAE) по всем маршрутам.
2. Максимизация общего опер. результата (pymoo минимизирует, поэтому оптимизируется отрицательный опер. результат).

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
from pymoo.core.repair import Repair
from pymoo.core.sampling import Sampling
from pymoo.optimize import minimize
from pymoo.termination import get_termination
from multiprocessing.pool import ThreadPool
import warnings

from simulation.multi_route import MultiRouteSimulation
from models.route import RouteConfig

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

# Динамически загружаем контрактное количество трамваев (tram_count) из конфигурационных файлов
MAX_PER_ROUTE_MAP = np.array([
    RouteConfig.from_json(ROUTE_PAIRS[r][0]).tram_count
    for r in ["20", "48", "55"]
])


# ─── Классы для операторов восстановления и генерации популяции ──────────────

class FleetSizeRepair(Repair):
    """
    Класс для восстановления ограничений (repair). Округляет переменные до целых
    чисел, клипирует по границам и обеспечивает, чтобы сумма значений не превышала лимит.
    """
    def __init__(self, n_max: int = N_MAX):
        super().__init__()
        self.n_max = n_max

    def _do(self, problem, X, **kwargs):
        # X может содержать вещественные значения после кроссовера/мутации
        X_rounded = np.round(X).astype(int)
        xl = problem.xl
        xu = problem.xu
        
        # Получаем лимит n_max из задачи, если он там настроен кастомно
        n_max = getattr(problem, "n_max", self.n_max)
        
        for i in range(len(X_rounded)):
            x = X_rounded[i]
            # Enforce individual bounds
            x = np.clip(x, xl, xu)
            
            # Enforce sum(x) <= n_max
            while sum(x) > n_max:
                available_indices = [idx for idx in range(len(x)) if x[idx] > xl[idx]]
                if not available_indices:
                    break
                # Выбираем маршрут с наибольшим парком для уменьшения
                idx_to_decrease = max(available_indices, key=lambda idx: x[idx])
                x[idx_to_decrease] -= 1
            X_rounded[i] = x
            
        return X_rounded.astype(float)


class FleetSampling(Sampling):
    """
    Генератор случайной начальной популяции, гарантирующий, что все сгенерированные
    особи удовлетворяют ограничению на общий объем парка n_max.
    """
    def _do(self, problem, n_samples, **kwargs):
        X = np.zeros((n_samples, problem.n_var), dtype=int)
        for i in range(problem.n_var):
            X[:, i] = np.random.randint(problem.xl[i], problem.xu[i] + 1, size=n_samples)
            
        # Восстанавливаем ограничение по сумме через FleetSizeRepair
        repair = FleetSizeRepair(n_max=getattr(problem, "n_max", N_MAX))
        return repair._do(problem, X, **kwargs)


# ─── Постановка задачи оптимизации в терминах pymoo ──────────────────────────

class TramFleetProblem(ElementwiseProblem):
    """
    Класс задачи оптимизации распределения парка трамваев.
    
    Переменные решения (x):
        Вектор [n_20, n_48, n_55] из целых чисел.
        Каждая переменная ограничена снизу 5, сверху — контрактным лимитом.
        
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
            xl=np.full(N_ROUTES, 5),            # Нижние границы (минимум 5 трамваев)
            xu=MAX_PER_ROUTE_MAP,               # Верхние границы (контрактные максимумы)
            vtype=int,                          # Тип переменных — целые числа
            runner=runner,
        )
        self.n_max = n_max
        import threading
        self._lock = threading.Lock()
        self.evaluated_history = []

    def _evaluate(self, x, out, *args, **kwargs):
        """
        Оценивает качество конкретного распределения трамваев с помощью запуска симуляции.

        :param x: Массив/вектор с количеством трамваев: [n_20, n_48, n_55].
        :param out: Словарь для записи вычисленных значений целей (F) и ограничений (G).
        """
        # Округляем значения и приводим к целым числам
        tram_counts = [int(np.round(v)) for v in x]
        
        # Гарантируем соблюдение индивидуальных границ
        for i in range(len(tram_counts)):
            tram_counts[i] = int(np.clip(tram_counts[i], self.xl[i], self.xu[i]))
            
        # Ремонтируем ограничение на общий размер парка (sum(x) <= n_max)
        while sum(tram_counts) > self.n_max:
            available_indices = [idx for idx in range(len(tram_counts)) if tram_counts[idx] > self.xl[idx]]
            if not available_indices:
                break
            # Уменьшаем на маршруте с наибольшим количеством трамваев
            idx_to_decrease = max(available_indices, key=lambda idx: tram_counts[idx])
            tram_counts[idx_to_decrease] -= 1
            
        # Записываем исправленные значения обратно в x, чтобы обновить особь в популяции
        for i in range(len(x)):
            x[i] = tram_counts[i]

        # Инициализируем многомаршрутную симуляцию с заданным распределением парка
        sim = MultiRouteSimulation.from_params(
            ROUTE_PAIRS,
            tram_counts=tram_counts,
            run_dir=None,
        )
        # Запускаем симуляцию в тихом режиме (без построения графиков и сохранения логов)
        sim.run(plot_graphs=False, save_logs=False)

        # Извлекаем агрегированные метрики симуляции из детальной статистики
        stats = sim.get_full_stats()
        headway_mae = stats["global"]["headway_mae_min"]
        marginal_profit = stats["global"]["marginal_profit"]

        # Формируем вектор целевых функций (прибыль берем с минусом для максимизации)
        out["F"] = np.array([headway_mae, -marginal_profit], dtype=float)
        # Формируем ограничение на вместимость парка (сумма ТС - лимит <= 0)
        out["G"] = np.array([sum(tram_counts) - self.n_max], dtype=float)

        # Сохраняем в историю в потокобезопасном режиме
        with self._lock:
            self.evaluated_history.append({
                "n_20": tram_counts[0],
                "n_48": tram_counts[1],
                "n_55": tram_counts[2],
                "headway_mae_min": headway_mae,
                "marginal_profit": marginal_profit,
            })

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

    # Инициализируем алгоритм NSGA-II с операторами для целочисленного кодирования и соблюдения лимита парка
    algorithm = NSGA2(
        pop_size=pop_size,                      # Число особей (вариантов распределения) в популяции
        sampling=FleetSampling(),              # Случайная начальная популяция с ограничением n_max
        crossover=SBX(prob=0.9, eta=15, vtype=float, repair=FleetSizeRepair(n_max=n_max)), # Скрещивание с ремонтом n_max
        mutation=PM(eta=20, vtype=float, repair=FleetSizeRepair(n_max=n_max)),             # Мутация с ремонтом n_max
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
    F = res.F   # Векторы целей: [headway_mae_min, -marginal_profit]

    if X is None or F is None:
        print("\n[Внимание] Не найдено ни одного допустимого решения.")
        return

    # Формируем DataFrame с результатами
    df = pd.DataFrame(
        np.hstack([X, F]),
        columns=["n_20", "n_48", "n_55", "headway_mae_min", "marginal_profit_neg"],
    )
    # Переводим прибыль обратно в положительные числа
    df["marginal_profit"] = -df["marginal_profit_neg"]
    df = df.drop(columns=["marginal_profit_neg"])
    # Добавляем суммарное количество трамваев для проверки ограничения
    df["total_trams"] = df[["n_20", "n_48", "n_55"]].sum(axis=1)

    # Сортируем решения по убыванию прибыли (для удобства чтения)
    df = df.sort_values("marginal_profit", ascending=False)
    csv_path = os.path.join(out_dir, "pareto_front.csv")
    df.to_csv(csv_path, index=False)
    
    print(f"\nPareto-фронт сохранён: {csv_path}")
    print(df.to_string(index=False))

    # Сохраняем все рассмотренные варианты (из истории оценок)
    problem = res.problem
    if hasattr(problem, "evaluated_history") and problem.evaluated_history:
        df_all = pd.DataFrame(problem.evaluated_history)
        # Удаляем дубликаты по конфигурации распределения парка
        df_all = df_all.drop_duplicates(subset=["n_20", "n_48", "n_55"])
        df_all["total_trams"] = df_all[["n_20", "n_48", "n_55"]].sum(axis=1)
        df_all = df_all.sort_values("marginal_profit", ascending=False)
        all_csv_path = os.path.join(out_dir, "all_evaluated.csv")
        df_all.to_csv(all_csv_path, index=False)
        print(f"Все рассмотренные варианты сохранены: {all_csv_path}")

    # Пытаемся построить графики Парето-фронта
    plots_dir = os.path.join(out_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    try:
        from plot_pareto import plot_pareto
        plot_pareto(csv_path=csv_path, out_dir=plots_dir, n_max=getattr(problem, "n_max", None))
    except Exception as e:
        print(f"Ошибка при построении графиков Парето: {e}")


# ─── Точка входа для локального тестирования оптимизатора ───────────────────

if __name__ == "__main__":
    # Тестовый прогон на 60 трамваев с увеличенными параметрами для получения богатого Парето-фронта
    run_nsga2(n_max=60, pop_size=30, n_gen=15)
