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

        self.last_fwd_dispatch_time: float = 0.0  # время последнего выпуска на fwd

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
        self.env.process(self._interval_aware_release())

    def _get_target_interval(self, t_min: float) -> float:
        """Целевой интервал (мин) для fwd-маршрута на момент t_min."""
        intervals = self.fwd.config.target_intervals
        if not intervals:
            return self._calc_fallback_interval()
        direction_key = list(intervals.keys())[0]
        direction_intervals = intervals[direction_key]
        hour = int(t_min // 60) % 24
        if hour < 7:
            hour_str = "before_07:00"
        elif hour == 23:
            hour_str = "23:00-24:00"
        else:
            hour_str = f"{hour:02d}:00-{hour+1:02d}:00"
        return direction_intervals.get(hour_str, self._calc_fallback_interval())

    def _calc_fallback_interval(self) -> float:
        """Fallback: оценка интервала = время_кругорейса / N."""
        fwd_km = sum(self.fwd.config.distances_list) / 1000
        bwd_km = sum(self.bwd.config.distances_list) / 1000
        eff_speed = max(
            self.fwd.config.flow_speed * (1 - DEFAULT_ROAD_LOAD_ESTIMATE),
            5.0,
        )
        travel_min = (fwd_km + bwd_km) / eff_speed * 60
        n_stops = len(self.fwd.config.stop_ids) + len(self.bwd.config.stop_ids)
        dwell_min = n_stops * 1.0
        turnaround = self.fwd.config.turnaround_time
        rest = max(self.fwd.config.min_rest_time, MIN_REST_TIME)
        round_trip = travel_min + dwell_min + turnaround + rest
        n = max(len(self.all_trams), 1)
        return round_trip / n

    def _interval_aware_release(self):
        """Утренний выпуск трамваев с целевым интервалом текущего часа."""
        for i, tram in enumerate(self.all_trams):
            if i > 0:
                interval = self._get_target_interval(self.env.now)
                target_time = self.env.now + interval
                yield self.env.timeout(interval)
            else:
                target_time = 0.0

            tram.target_release_time = target_time
            tram.actual_release_time = self.env.now
            self.last_fwd_dispatch_time = self.env.now
            log.info(
                f"[{self.env.now:.1f}] [TramPair {self.route_num}] "
                f"Утренний выпуск #{tram.tram_id}, "
                f"интервал={self._get_target_interval(self.env.now):.0f} мин"
            )
            yield self.pool.put(tram)

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
        """bwd завершён → отдых → ожидание целевого интервала → депо."""
        rest_time = max(self.fwd.config.min_rest_time, MIN_REST_TIME)
        while True:
            tram = yield self.bwd_done.get()
            log.info(
                f"[{self.env.now:.1f}] Трамвай #{tram.tram_id} "
                f"в депо, отдых {rest_time:.0f} мин"
            )
            yield self.env.timeout(rest_time)

            # Ждём до целевого интервала с момента последнего выпуска
            target_interval = self._get_target_interval(self.env.now)
            next_dispatch = self.last_fwd_dispatch_time + target_interval
            wait = next_dispatch - self.env.now

            if wait > 0:
                log.info(
                    f"[{self.env.now:.1f}] Трамвай #{tram.tram_id} "
                    f"ожидает {wait:.1f} мин до целевого интервала "
                    f"({target_interval:.0f} мин)"
                )
                yield self.env.timeout(wait)
            else:
                log.info(
                    f"[{self.env.now:.1f}] Трамвай #{tram.tram_id} "
                    f"опоздание на {-wait:.1f} мин — выпуск сразу"
                )

            tram.target_release_time = next_dispatch
            tram.actual_release_time = self.env.now
            self.last_fwd_dispatch_time = self.env.now
            log.info(
                f"[{self.env.now:.1f}] Трамвай #{tram.tram_id} "
                f"выпущен на линию"
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
            os.makedirs(os.path.join(run_dir, "logs"),  exist_ok=True)
            os.makedirs(os.path.join(run_dir, "plots"), exist_ok=True)
        else:
            self.run_dir = self._create_run_directory()

        # ── Загрузка экономических нормативов из Excel ─────────────────────────
        self.route_economics = load_route_economics()

        # ── Загрузка конфигурации штрафов ──────────────────────────────────────
        import json
        penalties_path = "configs/penalties_config.json"
        if os.path.exists(penalties_path):
            try:
                with open(penalties_path, "r", encoding="utf-8") as f:
                    self.penalties_config = json.load(f)
            except Exception as e:
                log.warning(f"Ошибка чтения {penalties_path}, используем дефолтные штрафы: {e}")
                self.penalties_config = self._default_penalties_dict()
        else:
            self.penalties_config = self._default_penalties_dict()
            try:
                os.makedirs(os.path.dirname(penalties_path), exist_ok=True)
                with open(penalties_path, "w", encoding="utf-8") as f:
                    json.dump(self.penalties_config, f, indent=2, ensure_ascii=False)
            except Exception as e:
                log.warning(f"Не удалось записать дефолтный конфиг штрафов: {e}")

        items = list(route_pairs.items())
        DEFAULT_PER_ROUTE = 30

        for i, (route_num, (fwd_file, bwd_file)) in enumerate(items):
            fwd_cfg = RouteConfig.from_json(fwd_file)
            bwd_cfg = RouteConfig.from_json(bwd_file)

            fwd_cfg.penalties_config = self.penalties_config
            bwd_cfg.penalties_config = self.penalties_config

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

    def _default_penalties_dict(self) -> dict:
        return {
            "release_early_tolerance_min": 1.0,
            "release_late_tolerance_min": 5.0,
            "midpoint_tolerance_min": 5.0,
            "failed_trips_threshold_pct": 15.0,
            "release_daily_penalty_rub": 1000.0,
            "midpoint_daily_penalty_rub": 1000.0
        }

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
        stats = self.get_full_stats()
        summary_text = self.generate_summary_text(stats)
        self._print_stats(summary_text)

        if save_logs and self.run_dir:
            # Сохранение текстового отчёта
            with open(os.path.join(self.run_dir, "simulation_summary.txt"), "w", encoding="utf-8") as f:
                f.write(summary_text)

            # Сохранение CSV отчёта
            import csv
            csv_path = os.path.join(self.run_dir, "simulation_summary.csv")
            with open(csv_path, "w", encoding="utf-8", newline="") as f:
                writer = csv.writer(f, delimiter=";")
                writer.writerow([
                    "Объект", "Пассажиры (расчет)", "Трамвай-км", "Рейсы", "Трамваев",
                    "Совокупный Доход", "Пассажирский Доход", "Контрактный Доход", 
                    "OpEx", "Маржинальная Прибыль", "Прибыль/км", "ROS %", "MAE интервалов",
                    "Невыполненные рейсы (выпуск)", "Рейсы с откл. середина", "Потерянный контракт", "Штрафы"
                ])
                
                for pair in self.pairs:
                    for direction in ("fwd", "bwd"):
                        rid = f"{pair.route_num}_{direction}"
                        rs = stats["routes"].get(rid)
                        if rs:
                            writer.writerow([
                                rid, f"{rs['passengers_estimated']:.0f}", f"{rs['tram_km']:.1f}", str(rs['total_trips']), str(rs['trams_count']),
                                f"{rs['revenue']:.0f}", f"{rs['passenger_revenue']:.0f}", f"{rs['contract_revenue']:.0f}",
                                f"{rs['opex']:.0f}", f"{rs['marginal_profit']:.0f}", f"{rs['profit_per_km']:.2f}",
                                f"{rs['ros_pct']:.1f}", f"{rs['headway_mae_min']:.2f}",
                                str(rs.get('failed_release_trips', 0)), str(rs.get('failed_midpoint_trips', 0)), f"{rs.get('lost_contract_revenue', 0.0):.0f}", "0"
                            ])
                            
                    rt = stats["route_totals"].get(pair.route_num)
                    if rt:
                        writer.writerow([
                            f"{pair.route_num}_общий", f"{rt['passengers_estimated']:.0f}", f"{rt['tram_km']:.1f}", str(rt['total_trips']), str(rt['trams_count']),
                            f"{rt['revenue']:.0f}", f"{rt['passenger_revenue']:.0f}", f"{rt['contract_revenue']:.0f}",
                            f"{rt['opex']:.0f}", f"{rt['marginal_profit']:.0f}", f"{rt['profit_per_km']:.2f}",
                            f"{rt['ros_pct']:.1f}", f"{rt['headway_mae_min']:.2f}",
                            str(rt.get('failed_release_trips', 0)), str(rt.get('failed_midpoint_trips', 0)), f"{rt.get('lost_contract_revenue', 0.0):.0f}", f"{rt.get('total_penalties', 0.0):.0f}"
                        ])
                        
                g = stats["global"]
                writer.writerow([
                    "Глобально", f"{g['total_passengers_est']:.0f}", f"{g['total_tram_km']:.1f}", str(g['total_trips']), str(g['total_trams']),
                    f"{g['total_revenue']:.0f}", f"{g['total_passenger_revenue']:.0f}", f"{g['total_contract_revenue']:.0f}",
                    f"{g['opex']:.0f}", f"{g['marginal_profit']:.0f}", f"{g['profit_per_km']:.2f}",
                    f"{g['ros_pct']:.1f}", f"{g['headway_mae_min']:.2f}",
                    str(g.get('failed_release_trips', 0)), str(g.get('failed_midpoint_trips', 0)), f"{g.get('lost_contract_revenue', 0.0):.0f}", f"{g.get('total_penalties', 0.0):.0f}"
                ])

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
        route_totals = {}
        
        for pair in self.pairs:
            # Сперва рассчитываем fwd и bwd индивидуально
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
                r_pax_rev = route.stats.total_passenger_revenue
                r_cnt_rev = route.stats.total_contract_revenue

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
                    "trams_count":                len(pair.all_trams),
                    "revenue":                    r_rev,
                    "passenger_revenue":          r_pax_rev,
                    "contract_revenue":           r_cnt_rev,
                    "headway_mae_min":            route_mae,
                    "opex":                       opex,
                    "marginal_profit":            marginal_profit,
                    "profit_per_km":              profit_per_km,
                    "ros_pct":                    ros,
                    "failed_release_trips":       route.stats.failed_release_trips,
                    "failed_midpoint_trips":      route.stats.failed_midpoint_trips,
                    "lost_contract_revenue":      route.stats.lost_contract_revenue,
                }

            # Затем рассчитываем агрегированную статистику по маршруту в целом
            fwd_s = routes_stats[pair.fwd.config.route_id]
            bwd_s = routes_stats[pair.bwd.config.route_id]
            
            tot_km = fwd_s["tram_km"] + bwd_s["tram_km"]
            tot_trips = fwd_s["total_trips"] + bwd_s["total_trips"]
            tot_rev = fwd_s["revenue"] + bwd_s["revenue"]
            tot_pax_rev = fwd_s["passenger_revenue"] + bwd_s["passenger_revenue"]
            tot_cnt_rev = fwd_s["contract_revenue"] + bwd_s["contract_revenue"]
            tot_opex = fwd_s["opex"] + bwd_s["opex"]
            tot_margin = tot_rev - tot_opex
            tot_pax = fwd_s["passengers_estimated"] + bwd_s["passengers_estimated"]
            
            pair_errors = [
                abs(d["headway_error_min"])
                for t in pair.all_trams
                for d in t.stats.schedule_deviations
                if d["route_id"].split("_")[0] == pair.route_num and d.get("headway_error_min") is not None
            ]
            pair_mae = sum(pair_errors) / len(pair_errors) if pair_errors else 0.0
            
            tot_failed_release = fwd_s["failed_release_trips"] + bwd_s["failed_release_trips"]
            tot_failed_midpoint = fwd_s["failed_midpoint_trips"] + bwd_s["failed_midpoint_trips"]
            tot_lost_contract = fwd_s["lost_contract_revenue"] + bwd_s["lost_contract_revenue"]
            
            # Daily penalties calculation
            release_penalty = 0.0
            midpoint_penalty = 0.0
            
            days = max(1.0, pair.fwd.config.simulation_hours / 24.0)
            p_config = self.penalties_config
            pct_threshold = p_config.get("failed_trips_threshold_pct", 15.0) / 100.0
            
            if tot_trips > 0:
                pct_release = tot_failed_release / tot_trips
                if pct_release > pct_threshold:
                    release_penalty = p_config.get("release_daily_penalty_rub", 1000.0) * days
                
                pct_midpoint = tot_failed_midpoint / tot_trips
                if pct_midpoint > pct_threshold:
                    midpoint_penalty = p_config.get("midpoint_daily_penalty_rub", 1000.0) * days
            
            tot_penalties = release_penalty + midpoint_penalty
            
            route_totals[pair.route_num] = {
                "passengers_estimated":       tot_pax,
                "tram_km":                    tot_km,
                "total_trips":                tot_trips,
                "trams_count":                len(pair.all_trams),
                "revenue":                    tot_rev,
                "passenger_revenue":          tot_pax_rev,
                "contract_revenue":           tot_cnt_rev,
                "headway_mae_min":            pair_mae,
                "opex":                       tot_opex,
                "release_penalty":            release_penalty,
                "midpoint_penalty":           midpoint_penalty,
                "total_penalties":            tot_penalties,
                "marginal_profit":            tot_margin - tot_penalties,
                "profit_per_km":              (tot_margin - tot_penalties) / tot_km if tot_km > 0 else 0.0,
                "ros_pct":                    ((tot_margin - tot_penalties) / tot_rev * 100) if tot_rev > 0 else 0.0,
                "failed_release_trips":       tot_failed_release,
                "failed_midpoint_trips":      tot_failed_midpoint,
                "lost_contract_revenue":      tot_lost_contract,
            }

        # ── Глобальные агрегаты ────────────────────────────────────────────
        g_total_trips = sum(rs["total_trips"] for rs in routes_stats.values())
        g_opex = sum(rs["opex"] for rs in routes_stats.values())
        g_passenger_revenue = sum(rs["passenger_revenue"] for rs in routes_stats.values())
        g_contract_revenue = sum(rs["contract_revenue"] for rs in routes_stats.values())
        g_total_trams = sum(len(p.all_trams) for p in self.pairs)
        
        g_failed_release = sum(rt["failed_release_trips"] for rt in route_totals.values())
        g_failed_midpoint = sum(rt["failed_midpoint_trips"] for rt in route_totals.values())
        g_lost_contract = sum(rt["lost_contract_revenue"] for rt in route_totals.values())
        g_release_penalty = sum(rt["release_penalty"] for rt in route_totals.values())
        g_midpoint_penalty = sum(rt["midpoint_penalty"] for rt in route_totals.values())
        g_total_penalties = g_release_penalty + g_midpoint_penalty
        
        g_marginal_profit = total_revenue - g_opex - g_total_penalties
        g_profit_per_km = g_marginal_profit / total_km if total_km > 0 else 0.0
        g_ros = (g_marginal_profit / total_revenue * 100) if total_revenue > 0 else 0.0

        return {
            "routes": routes_stats,
            "route_totals": route_totals,
            "global": {
                "total_tram_km":        total_km,
                "total_trips":          g_total_trips,
                "headway_mae_min":      headway_mae,
                "total_revenue":        total_revenue,
                "total_passenger_revenue": g_passenger_revenue,
                "total_contract_revenue":  g_contract_revenue,
                "total_passengers_est": total_pax_est,
                "unique_stops":         len(self.shared_stops),
                "opex":                 g_opex,
                "release_penalty":      g_release_penalty,
                "midpoint_penalty":     g_midpoint_penalty,
                "total_penalties":      g_total_penalties,
                "marginal_profit":      g_marginal_profit,
                "profit_per_km":        g_profit_per_km,
                "ros_pct":              g_ros,
                "total_trams":          g_total_trams,
                "failed_release_trips": g_failed_release,
                "failed_midpoint_trips": g_failed_midpoint,
                "lost_contract_revenue": g_lost_contract,
            },
        }

    def generate_summary_text(self, stats: dict) -> str:
        lines = []
        lines.append(f"\n{'='*60}")
        lines.append("РЕЗУЛЬТАТЫ МУЛЬТИМАРШРУТНОЙ СИМУЛЯЦИИ")
        lines.append("="*60)
        
        for pair in self.pairs:
            for direction in ("fwd", "bwd"):
                rid = f"{pair.route_num}_{direction}"
                rs = stats["routes"].get(rid)
                if rs:
                    lines.append(f"\nМаршрут {rid}:")
                    lines.append(f"  • Пассажиры (расчёт):  {rs['passengers_estimated']:.0f}")
                    lines.append(f"  • Трамвай-км:          {rs['tram_km']:.1f}")
                    lines.append(f"  • Рейсов:             {rs['total_trips']}")
                    lines.append(f"  • Трамваев на маршруте:  {rs['trams_count']}")
                    lines.append(f"  • Совокупный Доход:    {rs['revenue']:.0f} руб.")
                    lines.append(f"    - Пассажиры:         {rs['passenger_revenue']:.0f} руб.")
                    lines.append(f"    - Контракт:          {rs['contract_revenue']:.0f} руб.")
                    lines.append(f"  • OpEx:                {rs['opex']:.0f} руб.")
                    lines.append(f"  • Марж. прибыль:      {rs['marginal_profit']:.0f} руб.")
                    lines.append(f"  • Прибыль/км:          {rs['profit_per_km']:.2f} руб.")
                    lines.append(f"  • ROS:                 {rs['ros_pct']:.1f}%")
                    lines.append(f"  • MAE интервалов:     {rs['headway_mae_min']:.2f} мин")
            
            rt = stats["route_totals"].get(pair.route_num)
            if rt:
                lines.append(f"\nМаршрут {pair.route_num} общий:")
                lines.append(f"  • Пассажиры (расчёт):  {rt['passengers_estimated']:.0f}")
                lines.append(f"  • Трамвай-км:          {rt['tram_km']:.1f}")
                lines.append(f"  • Рейсов:             {rt['total_trips']}")
                lines.append(f"  • Невыполненных рейсов (выпуск): {rt['failed_release_trips']} (потери контракта: {rt['lost_contract_revenue']:.0f} руб.)")
                lines.append(f"  • Рейсов с отклонением на серединной: {rt['failed_midpoint_trips']}")
                lines.append(f"  • Трамваев на маршруте:  {rt['trams_count']}")
                lines.append(f"  • Совокупный Доход:    {rt['revenue']:.0f} руб.")
                lines.append(f"    - Пассажиры:         {rt['passenger_revenue']:.0f} руб.")
                lines.append(f"    - Контракт:          {rt['contract_revenue']:.0f} руб.")
                lines.append(f"  • OpEx:                {rt['opex']:.0f} руб.")
                if rt.get('total_penalties', 0) > 0:
                    lines.append(f"  • Штрафы за несоблюдение выпуска: {rt['release_penalty']:.0f} руб.")
                    lines.append(f"  • Штрафы за серединные отклонения: {rt['midpoint_penalty']:.0f} руб.")
                    lines.append(f"  • Всего штрафов:      {rt['total_penalties']:.0f} руб.")
                lines.append(f"  • Марж. прибыль:      {rt['marginal_profit']:.0f} руб.")
                lines.append(f"  • Прибыль/км:          {rt['profit_per_km']:.2f} руб.")
                lines.append(f"  • ROS:                 {rt['ros_pct']:.1f}%")
                lines.append(f"  • MAE интервалов:     {rt['headway_mae_min']:.2f} мин")
                
        g = stats["global"]
        lines.append(f"\nГлобально:")
        lines.append(f"  • Всего трамвай-км:   {g['total_tram_km']:.1f}")
        lines.append(f"  • Всего рейсов:      {g['total_trips']}")
        lines.append(f"  • Всего невыполненных рейсов: {g['failed_release_trips']} (всего потери контракта: {g['lost_contract_revenue']:.0f} руб.)")
        lines.append(f"  • Всего рейсов с отклонением на серединной: {g['failed_midpoint_trips']}")
        lines.append(f"  • Всего трамваев:      {g['total_trams']}")
        lines.append(f"  • Всего доход:       {g['total_revenue']:.0f} руб.")
        lines.append(f"    - Пассажиры:       {g['total_passenger_revenue']:.0f} руб.")
        lines.append(f"    - Контракт:        {g['total_contract_revenue']:.0f} руб.")
        lines.append(f"  • Всего OpEx:        {g['opex']:.0f} руб.")
        if g.get('total_penalties', 0) > 0:
            lines.append(f"  • Всего штрафов за выпуск: {g['release_penalty']:.0f} руб.")
            lines.append(f"  • Всего штрафов за серединные: {g['midpoint_penalty']:.0f} руб.")
            lines.append(f"  • Всего штрафов:      {g['total_penalties']:.0f} руб.")
        lines.append(f"  • Марж. прибыль:     {g['marginal_profit']:.0f} руб.")
        lines.append(f"  • Прибыль/км:        {g['profit_per_km']:.2f} руб.")
        lines.append(f"  • ROS:               {g['ros_pct']:.1f}%")
        lines.append(f"  • Пассажиры (расч.): {g['total_passengers_est']:.0f}")
        lines.append(f"  • Unique stops:      {g['unique_stops']}")
        lines.append(f"  • MAE интервалов:     {g['headway_mae_min']:.2f} мин")
        lines.append(f"{'='*60}\n")
        
        return "\n".join(lines)

    def _print_stats(self, summary_text: Optional[str] = None):
        if summary_text is None:
            stats = self.get_full_stats()
            summary_text = self.generate_summary_text(stats)
        results_log.info(summary_text)

