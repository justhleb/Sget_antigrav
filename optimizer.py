# optimizer.py
from __future__ import annotations

import logging
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pymoo.core.problem import ElementwiseProblem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.operators.sampling.rnd import IntegerRandomSampling
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.repair.rounding import RoundingRepair
from pymoo.optimize import minimize
from pymoo.termination import get_termination
# removed StarmapParallelization
from multiprocessing.pool import ThreadPool
import warnings

from simulation.multi_route import MultiRouteSimulation

warnings.filterwarnings("ignore")

# Особый режим логирования для оптимизатора (отключаем лишний вывод)
logging.basicConfig(level=logging.ERROR, force=True)
# ─── конфиги маршрутов ────────────────────────────────────────────────────────
ROUTE_PAIRS = {
    "20": ("configs/route_20_fwd_config.json", "configs/route_20_bwd_config.json"),
    "48": ("configs/route_48_fwd_config.json", "configs/route_48_bwd_config.json"),
    "55": ("configs/route_55_fwd_config.json", "configs/route_55_bwd_config.json"),
}

N_ROUTES = len(ROUTE_PAIRS)   # 3
N_MAX    = 45                  # бюджет парка 


# ─── задача оптимизации ───────────────────────────────────────────────────────

class TramFleetProblem(ElementwiseProblem):
    """
    Переменные:   x = [n_20, n_48, n_55]  — целые, диапазон [5, 30]
    Цели:         F = [total_tram_km, headway_mae]  — минимизируем обе
    Ограничения:  G = [sum(x) - N_MAX]  ≤ 0
    """

    def __init__(self, n_max: int = N_MAX, runner=None):
        super().__init__(
            n_var=N_ROUTES,
            n_obj=3,
            n_ieq_constr=1,
            xl=np.full(N_ROUTES, 5),
            xu=np.full(N_ROUTES, 30),
            vtype=int,
            runner=runner,
        )
        self.n_max = n_max

    def _evaluate(self, x, out, *args, **kwargs):
        tram_counts = [int(v) for v in x]

        # silent=True — не создаём папки на диске
        sim = MultiRouteSimulation.from_params(
            ROUTE_PAIRS,
            tram_counts=tram_counts,
            run_dir=None,   
        )
        sim.run(plot_graphs=False, save_logs=False)

        _, total_km, headway_mae, total_served = sim.get_objectives()

        out["F"] = np.array([-total_km, headway_mae, -total_served], dtype=float)
        out["G"] = np.array([sum(tram_counts) - self.n_max], dtype=float)
        
        del sim


# ─── запуск NSGA-II ───────────────────────────────────────────────────────────

def run_nsga2(
    n_max:    int = N_MAX,
    pop_size: int = 25,
    n_gen:    int = 5,
    seed:     int = 42,
    out_dir:  str = "outputs/nsga2",
) -> tuple:
    from datetime import datetime
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = os.path.join(out_dir, f"run_{ts}")
    os.makedirs(run_dir, exist_ok=True)

    n_threads = 8  # Можно поменять в зависимости от процессора
    pool = ThreadPool(n_threads)
    runner = pool.starmap
    
    problem = TramFleetProblem(n_max=n_max, runner=runner)

    algorithm = NSGA2(
        pop_size=pop_size,
        sampling=IntegerRandomSampling(),
        crossover=SBX(prob=0.9, eta=15, vtype=float, repair=RoundingRepair()),
        mutation=PM(eta=20, vtype=float, repair=RoundingRepair()),
        eliminate_duplicates=True,
    )

    termination = get_termination("n_gen", n_gen)

    print(f"Запуск NSGA-II: pop={pop_size}, gen={n_gen}, n_max={n_max}")
    print(f"Всего evaluations: ~{pop_size * n_gen}")

    res = minimize(
        problem,
        algorithm,
        termination,
        seed=seed,
        verbose=True,
        save_history=False,
    )

    pool.close()
    pool.join()

    # ─── сохраняем Pareto-фронт ───────────────────────────────────────────────
    _save_results(res, run_dir)

    return res


def _save_results(res, out_dir: str):
    X = res.X   # переменные: [n_20, n_48, n_55]
    F = res.F   # цели:       [total_km, headway_mae]

    if X is None or F is None:
        print("\n[Внимание] Не найдено ни одного допустимого решения (возможно, слишком строгие ограничения, например n_max).")
        return

    # CSV
    df = pd.DataFrame(
        np.hstack([X, F]),
        columns=["n_20", "n_48", "n_55", "total_tram_km_neg", "headway_mae_min", "total_served_neg"],
    )
    df["total_tram_km"] = -df["total_tram_km_neg"]
    df["total_served"]  = -df["total_served_neg"]
    df = df.drop(columns=["total_tram_km_neg", "total_served_neg"])   # возвращаем знак обратно
    df["total_trams"] = df[["n_20", "n_48", "n_55"]].sum(axis=1)

    df = df.sort_values("total_tram_km")
    csv_path = os.path.join(out_dir, "pareto_front.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nPareto-фронт сохранён: {csv_path}")
    print(df.to_string(index=False))

    plots_dir = os.path.join(out_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    try:
        from plot_pareto import plot_pareto
        plot_pareto(csv_path=csv_path, out_dir=plots_dir)
    except Exception as e:
        print(f"Ошибка при построении графиков: {e}")


# ─── точка входа ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    run_nsga2(n_max=45, pop_size=25, n_gen=5)
