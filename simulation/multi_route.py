"""
MultiRouteSimulation — оркестратор нескольких маршрутов в едином env.

Использование:
    sim = MultiRouteSimulation({
        "20": ("configs/route_20_fwd_config.json", "configs/route_20_bwd_config.json"),
        "48": ("configs/route_48_fwd_config.json", "configs/route_48_bwd_config.json"),
    })
    sim.run()

Для NSGA-II:
    sim = MultiRouteSimulation.from_params(route_pairs, tram_counts=[30, 30, 30])
    sim.run(plot_graphs=False, save_logs=False)
    cost = sim.get_objectives()
"""
from __future__ import annotations

import logging
import os
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import simpy

from models.stop import Stop
from models.route import Route, RouteConfig
from models.tram import Tram
from logger import TramLogger
from visualization import TramVisualization
from data_sensors.excel_parser import load_route_economics

log = logging.getLogger(__name__)
results_log = logging.getLogger("simulation.results")

OUTPUT_DIR    = "outputs"
MIN_REST_TIME = 15.0
DEFAULT_ROAD_LOAD_ESTIMATE = 0.4  # для оценки интервала начального выпуска


class TramPair:
    """
    Пара fwd/bwd маршрутов с тремя Store:
      pool     — свободные трамваи (депо)
      fwd_done — трамваи завершившие fwd, ждут разворота
      bwd_done — трамваи завершившие bwd, ждут отдыха
    """

    def __init__(
        self,
        route_num: str,
        fwd_config: RouteConfig,
        bwd_config: RouteConfig,
        env: simpy.Environment,
        shared_stops: Dict[int, Stop],
        tram_count: int,
        tram_id_offset: int,
        lightweight_mode: bool = False,
    ):
        self.route_num = route_num
        self.env = env

        self.pool     = simpy.Store(env)
        self.fwd_done = simpy.Store(env)
        self.bwd_done = simpy.Store(env)

        self.fwd = Route(fwd_config, env, shared_stops,
                         available_trams=self.pool,
                         done_store=self.fwd_done)
        self.bwd = Route(bwd_config, env, shared_stops,
                         available_trams=self.fwd_done,
                         done_store=self.bwd_done)

        self.all_trams: List[Tram] = []
        self._spawn_trams(tram_count, tram_id_offset, fwd_config.tram_capacity, lightweight_mode)

    def _spawn_trams(self, count: int, id_offset: int, capacity: int, lightweight_mode: bool):
        for i in range(count):
            tram_id = id_offset + i + 1
            tram = Tram(tram_id, self.route_num, capacity, lightweight_mode=lightweight_mode)
            self.all_trams.append(tram)
        log.info(
            f"[TramPair {self.route_num}] Парк: {count} трамваев "
            f"(id {id_offset + 1}..{id_offset + count})"
        )

    def start(self):
        self.fwd.start()
        self.bwd.start()
        self.env.process(self._turnaround_process())
        self.env.process(self._rest_process())
        self.env.process(self._staggered_release())

    def _calc_dispatch_interval(self) -> float:
        """Оценка интервала между выпусками = время_кругорейса / N."""
        fwd_km = sum(self.fwd.config.distances.values()) / 1000
        bwd_km = sum(self.bwd.config.distances.values()) / 1000
        eff_speed = max(
            self.fwd.config.flow_speed * (1 - DEFAULT_ROAD_LOAD_ESTIMATE),
            5.0,
        )
        travel_min = (fwd_km + bwd_km) / eff_speed * 60
        n_stops = len(self.fwd.config.stop_ids) + len(self.bwd.config.stop_ids)
        dwell_min = n_stops * 1.0  # ~1 мин на остановку
        turnaround = self.fwd.config.turnaround_time
        rest = max(self.fwd.config.min_rest_time, MIN_REST_TIME)
        round_trip = travel_min + dwell_min + turnaround + rest
        n = max(len(self.all_trams), 1)
        interval = round_trip / n
        log.info(
            f"[TramPair {self.route_num}] Оценка кругорейса: {round_trip:.0f} мин, "
            f"интервал выпуска: {interval:.1f} мин"
        )
        return interval

    def _staggered_release(self):
        """Выпускает трамваи из депо с равным интервалом."""
        interval = self._calc_dispatch_interval()
        for i, tram in enumerate(self.all_trams):
            yield self.pool.put(tram)
            if i < len(self.all_trams) - 1:
                yield self.env.timeout(interval)

    def _turnaround_process(self):
        """fwd завершён → разворот → трамвай доступен для bwd."""
        turnaround = self.fwd.config.turnaround_time
        while True:
            tram = yield self.fwd_done.get()
            log.info(
                f"[{self.env.now:.1f}] Трамвай #{tram.tram_id} "
                f"разворачивается ({turnaround} мин)"
            )
            yield self.env.timeout(turnaround)
            yield self.fwd_done.put(tram)

    def _rest_process(self):
        """bwd завершён → отдых водителя → трамвай обратно в депо."""
        rest_time = max(self.fwd.config.min_rest_time, MIN_REST_TIME)
        while True:
            tram = yield self.bwd_done.get()
            log.info(
                f"[{self.env.now:.1f}] Трамвай #{tram.tram_id} "
                f"в депо, отдых {rest_time:.0f} мин"
            )
            yield self.env.timeout(rest_time)
            log.info(
                f"[{self.env.now:.1f}] Трамвай #{tram.tram_id} "
                f"готов к новому рейсу"
            )
            yield self.pool.put(tram)


class MultiRouteSimulation:

    def __init__(
        self,
        route_pairs: Dict[str, Tuple[str, str]],
        tram_counts: Optional[List[int]] = None,
        run_dir: Optional[str] = None,
        silent: bool = False,   # True — оптимизатор, папка не создаётся
        lightweight_mode: bool = False, # True — не сохраняем историю событий
    ):
        self.env = simpy.Environment()
        self.shared_stops: Dict[int, Stop] = {}
        self.pairs: List[TramPair] = []

        if silent:
            self.run_dir = None
        elif run_dir is not None:
            self.run_dir = run_dir
        else:
            self.run_dir = self._create_run_directory()

        # ── Загрузка экономических нормативов из Excel ─────────────────────────
        self.route_economics = load_route_economics()

        items = list(route_pairs.items())
        DEFAULT_PER_ROUTE = 30

        for i, (route_num, (fwd_file, bwd_file)) in enumerate(items):
            fwd_cfg = RouteConfig.from_json(fwd_file)
            bwd_cfg = RouteConfig.from_json(bwd_file)

            # Инжектим экономические нормативы в конфиг маршрута
            econ = self.route_economics.get(route_num)
            if econ is not None:
                fwd_cfg.revenue_per_km    = econ.mean_revenue_per_km
                fwd_cfg.passengers_per_km = econ.mean_passengers_per_km
                bwd_cfg.revenue_per_km    = econ.mean_revenue_per_km
                bwd_cfg.passengers_per_km = econ.mean_passengers_per_km
                log.info(
                    f"Маршрут {route_num}: "
                    f"revenue/km={econ.mean_revenue_per_km:.2f}, "
                    f"pax/km={econ.mean_passengers_per_km:.2f}"
                )
            else:
                log.warning(
                    f"Маршрут {route_num}: нет экономических данных — "
                    f"revenue_per_km и passengers_per_km будут = 0"
                )

            self._register_stops(fwd_cfg)
            self._register_stops(bwd_cfg)

            count  = tram_counts[i] if tram_counts else DEFAULT_PER_ROUTE
            offset = sum(tram_counts[:i]) if tram_counts else i * DEFAULT_PER_ROUTE

            pair = TramPair(
                route_num=route_num,
                fwd_config=fwd_cfg,
                bwd_config=bwd_cfg,
                env=self.env,
                shared_stops=self.shared_stops,
                tram_count=count,
                tram_id_offset=offset,
                lightweight_mode=lightweight_mode,
            )
            self.pairs.append(pair)

        total = sum(tram_counts) if tram_counts else len(items) * DEFAULT_PER_ROUTE
        log.info(
            f"MultiRouteSimulation: {len(self.pairs)} маршрута, "
            f"{total} трамваев всего, "
            f"{len(self.shared_stops)} уникальных остановок"
        )

        # ── Добавление файлового лога для всех (включая служебные) сообщений прогона ──────
        if self.run_dir:
            logs_dir = os.path.join(self.run_dir, "logs")
            log_file_path = os.path.join(logs_dir, "simulation.log")
            self._file_handler = logging.FileHandler(log_file_path, encoding="utf-8")
            self._file_handler.setLevel(logging.INFO)
            self._file_handler.setFormatter(logging.Formatter("%(message)s"))
            logging.getLogger().addHandler(self._file_handler)

    # ── Фабричный метод для NSGA-II ───────────────────────────────────────────

    @classmethod
    def from_params(
        cls,
        route_pairs: Dict[str, Tuple[str, str]],
        tram_counts: List[int],
        run_dir: Optional[str] = None,
    ) -> "MultiRouteSimulation":
        """
        Создаёт симуляцию для NSGA-II — без создания папок на диске.
        """
        return cls(route_pairs, tram_counts=tram_counts,
                   run_dir=run_dir, silent=True, lightweight_mode=True)

    # ── Регистрация остановок ─────────────────────────────────────────────────

    def _register_stops(self, cfg: RouteConfig):
        for stop_id in cfg.stop_ids:
            if stop_id not in self.shared_stops:
                stop = Stop(stop_id, self.env)
                # Если папок не создаем — значит оптимизатор (silent), включим легкий режим
                if self.run_dir is None:
                    stop.lightweight_mode = True
                self.shared_stops[stop_id] = stop

    # ── Вспомогательные ───────────────────────────────────────────────────────

    def _create_run_directory(self) -> str:
        ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        run_dir = os.path.join(OUTPUT_DIR, f"run_{ts}")
        os.makedirs(os.path.join(run_dir, "logs"),  exist_ok=True)
        os.makedirs(os.path.join(run_dir, "plots"), exist_ok=True)
        log.info(f"Результаты: {run_dir}/")
        return run_dir

    def _max_hours(self) -> int:
        return max(
            max(p.fwd.config.simulation_hours, p.bwd.config.simulation_hours)
            for p in self.pairs
        )

    def _all_trams(self) -> List[Tram]:
        return [t for p in self.pairs for t in p.all_trams]

    # ── Запуск ────────────────────────────────────────────────────────────────

    def run(self, plot_graphs: bool = True, save_logs: bool = True):
        for pair in self.pairs:
            pair.start()

        self.env.run(until=self._max_hours() * 60)
        self._print_stats()

        if save_logs and self.run_dir:
            logs_dir = os.path.join(self.run_dir, "logs")
            for pair in self.pairs:
                route_logs = os.path.join(logs_dir, pair.route_num)
                tl = TramLogger(output_dir=route_logs)
                trams = {t.tram_id: t for t in pair.all_trams}
                tl.save_all_trams(trams, route_id=pair.route_num)
                tl.save_schedule_deviations(trams, route_id=pair.route_num)

        if plot_graphs and self.run_dir:
            plots_dir = os.path.join(self.run_dir, "plots")
            for pair in self.pairs:
                route_plots = os.path.join(plots_dir, pair.route_num)
                os.makedirs(route_plots, exist_ok=True)

                combined_stops = {
                    sid: self.shared_stops[sid]
                    for route in (pair.fwd, pair.bwd)
                    for sid in route.config.stop_ids
                    if sid in self.shared_stops
                }
                viz = TramVisualization(
                    combined_stops,
                    max(pair.fwd.config.simulation_hours,
                        pair.bwd.config.simulation_hours),
                    route_id=pair.route_num,
                )
                trams = {t.tram_id: t for t in pair.all_trams}

                viz.create_all_plots(
                    trams=trams,
                    output_dir=route_plots,
                )

        if plot_graphs and self.run_dir:
            from visualization import plot_global_financial_summary
            stats = self.get_full_stats()
            plots_dir = os.path.join(self.run_dir, "plots")
            global_plot_file = os.path.join(plots_dir, "global_financial_summary.png")
            plot_global_financial_summary(stats, global_plot_file)

        # Закрываем и удаляем FileHandler, чтобы логи не дублировались/не утекали в будущем
        if hasattr(self, "_file_handler"):
            logging.getLogger().removeHandler(self._file_handler)
            self._file_handler.close()

    # ── Метрики ───────────────────────────────────────────────────────────────

    def get_objectives(self) -> Tuple[float, float, float, float]:
        """
        Возвращает (total_tram_km, headway_mae, total_revenue, total_passengers_est).

        total_tram_km      — максимизируем (транспортная работа по контракту)
        headway_mae        — минимизируем (точность поддержания интервала)
        total_revenue      — максимизируем (чистый доход, руб.)
        total_pax_est      — расчётные пассажиры
        """
        total_km = sum(
            r.stats.total_tram_km
            for p in self.pairs
            for r in (p.fwd, p.bwd)
        )

        total_revenue = sum(
            r.stats.total_revenue
            for p in self.pairs
            for r in (p.fwd, p.bwd)
        )

        total_pax_est = sum(
            r.stats.total_passengers_estimated
            for p in self.pairs
            for r in (p.fwd, p.bwd)
        )

        all_errors = [
            abs(d["headway_error_min"])
            for p in self.pairs
            for t in p.all_trams
            for d in t.stats.schedule_deviations
            if d.get("headway_error_min") is not None
        ]
        headway_mae = sum(all_errors) / len(all_errors) if all_errors else 0.0

        return total_km, headway_mae, total_revenue, total_pax_est

    def get_full_stats(self) -> dict:
        total_km, headway_mae, total_revenue, total_pax_est = self.get_objectives()
        routes_stats = {}
        for pair in self.pairs:
            for route in (pair.fwd, pair.bwd):
                cfg = route.config
                route_errors = [
                    abs(d["headway_error_min"])
                    for t in pair.all_trams
                    for d in t.stats.schedule_deviations
                    if d["route_id"] == cfg.route_id and d.get("headway_error_min") is not None
                ]
                route_mae = sum(route_errors) / len(route_errors) if route_errors else 0.0

                r_km = route.stats.total_tram_km
                r_trips = route.stats.total_trips
                r_rev = route.stats.total_revenue

                cost_per_km = cfg.energy_per_km + cfg.maintenance_per_km + cfg.depreciation_per_km
                cost_per_trip = cfg.payroll_per_trip + cfg.maintenance_fixed_per_trip
                opex = (r_km * cost_per_km) + (r_trips * cost_per_trip)
                marginal_profit = r_rev - opex
                profit_per_km = marginal_profit / r_km if r_km > 0 else 0.0
                ros = (marginal_profit / r_rev * 100) if r_rev > 0 else 0.0

                routes_stats[cfg.route_id] = {
                    "passengers_estimated":       route.stats.total_passengers_estimated,
                    "tram_km":                    r_km,
                    "total_trips":                r_trips,
                    "revenue":                    r_rev,
                    "headway_mae_min":            route_mae,
                    "opex":                       opex,
                    "marginal_profit":            marginal_profit,
                    "profit_per_km":              profit_per_km,
                    "ros_pct":                    ros,
                }

        # ── Глобальные агрегаты ────────────────────────────────────────────
        g_total_trips = sum(rs["total_trips"] for rs in routes_stats.values())
        g_opex = sum(rs["opex"] for rs in routes_stats.values())
        g_marginal_profit = total_revenue - g_opex
        g_profit_per_km = g_marginal_profit / total_km if total_km > 0 else 0.0
        g_ros = (g_marginal_profit / total_revenue * 100) if total_revenue > 0 else 0.0

        return {
            "routes": routes_stats,
            "global": {
                "total_tram_km":        total_km,
                "total_trips":          g_total_trips,
                "headway_mae_min":      headway_mae,
                "total_revenue":        total_revenue,
                "total_passengers_est": total_pax_est,
                "unique_stops":         len(self.shared_stops),
                "opex":                 g_opex,
                "marginal_profit":      g_marginal_profit,
                "profit_per_km":        g_profit_per_km,
                "ros_pct":              g_ros,
            },
        }

    def _print_stats(self):
        results_log.info(f"\n{'='*60}")
        results_log.info("РЕЗУЛЬТАТЫ МУЛЬТИМАРШРУТНОЙ СИМУЛЯЦИИ")
        results_log.info(f"{'='*60}")
        stats = self.get_full_stats()
        for route_id, rs in stats["routes"].items():
            results_log.info(f"\nМаршрут {route_id}:")
            results_log.info(f"  • Пассажиры (расчёт):  {rs['passengers_estimated']:.0f}")
            results_log.info(f"  • Трамвай-км:          {rs['tram_km']:.1f}")
            results_log.info(f"  • Рейсов:             {rs['total_trips']}")
            results_log.info(f"  • Доход:               {rs['revenue']:.0f} руб.")
            results_log.info(f"  • OpEx:                {rs['opex']:.0f} руб.")
            results_log.info(f"  • Марж. прибыль:      {rs['marginal_profit']:.0f} руб.")
            results_log.info(f"  • Прибыль/км:          {rs['profit_per_km']:.2f} руб.")
            results_log.info(f"  • ROS:                 {rs['ros_pct']:.1f}%")
            results_log.info(f"  • MAE интервалов:     {rs['headway_mae_min']:.2f} мин")
        g = stats["global"]
        results_log.info(f"\nГлобально:")
        results_log.info(f"  • Всего трамвай-км:   {g['total_tram_km']:.1f}")
        results_log.info(f"  • Всего рейсов:      {g['total_trips']}")
        results_log.info(f"  • Всего доход:       {g['total_revenue']:.0f} руб.")
        results_log.info(f"  • Всего OpEx:        {g['opex']:.0f} руб.")
        results_log.info(f"  • Марж. прибыль:     {g['marginal_profit']:.0f} руб.")
        results_log.info(f"  • Прибыль/км:        {g['profit_per_km']:.2f} руб.")
        results_log.info(f"  • ROS:               {g['ros_pct']:.1f}%")
        results_log.info(f"  • Пассажиры (расч.): {g['total_passengers_est']:.0f}")
        results_log.info(f"  • Уникальных ост.:    {g['unique_stops']}")
        results_log.info(f"  • MAE интервалов:     {g['headway_mae_min']:.2f} мин")
        results_log.info(f"{'='*60}\n")

