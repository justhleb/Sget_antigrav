"""
RouteConfig — чистый датакласс конфига маршрута (загрузка из JSON).
Route       — SimPy-процесс одного прогона маршрута (только fwd ИЛИ только bwd).
              Разворот и полный цикл fwd→bwd управляется из MultiRoute.
"""
from __future__ import annotations

import json
import logging
import random
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import simpy

from models.stop import Stop, StopEvent
from models.tram import Tram
from constants import DEFAULT_STOP_DWELL_MIN

log = logging.getLogger(__name__)

# ── Дефолты ───────────────────────────────────────────────────────────────────
DEFAULT_TURNAROUND  = 2.0
DEFAULT_TARGET_UTIL = 0.75
DEFAULT_ROAD_LOAD   = 0.50
MIN_SPEED_KMH       = 5.0
SPEED_VARIATION     = 0.05


@dataclass
class RouteStats:
    route_id: str
    total_passengers_served: int = 0
    total_tram_km: float = 0.0
    total_passenger_km: float = 0.0
    total_passenger_revenue: float = 0.0 # руб. (доход от пассажиров)
    total_contract_revenue: float = 0.0  # руб. (доход от контракта)
    total_revenue: float = 0.0          # руб. (совокупный доход)
    total_passengers_estimated: float = 0.0  # расчетные пассажиры (пробег * pax_per_km)
    total_trips: int = 0                # количество завершённых рейсов
    utilization_deviations: List[float] = field(default_factory=list)




@dataclass
class RouteConfig:
    route_id: str
    stop_ids: List[int]
    tram_capacity: int
    flow_speed: float
    peak_stop_index: int
    simulation_hours: int
    distances: Dict[int, float]
    road_loads: Dict[int, float]
    distances_list: List[float] = field(default_factory=list)
    depot_to_first_stop: float = 8.0
    min_rest_time: float = 15.0
    turnaround_time: float = DEFAULT_TURNAROUND
    acceleration_time: float = 0.5
    stop_time: float = 1.0
    target_utilization: float = DEFAULT_TARGET_UTIL
    random_seed: Optional[int] = None
    target_intervals: Optional[Dict[str, Dict[str, int]]] = None
    # ── Экономические нормативы (из Excel) ────────────────────────────────
    revenue_per_km: float = 0.0       # руб. средний доход на 1 км пути
    passengers_per_km: float = 0.0    # чел. среднее кол-во пасс. на 1 км
    # ── Расходные параметры (из JSON-конфигов) ────────────────────────────
    contract_revenue_per_km: float = 529.5  # руб. за км от исполнения контракта
    energy_per_km: float = 0.0              # энергия на км, руб.
    maintenance_per_km: float = 0.0         # ТОиР на км (пробежный), руб.
    depreciation_per_km: float = 0.0        # амортизация на км, руб.
    payroll_per_trip: float = 0.0            # ФОТ на рейс, руб.
    maintenance_fixed_per_trip: float = 0.0  # ТОиР на рейс (фикс.), руб.

    @property
    def stop_number(self) -> int:
        return len(self.stop_ids)

    @classmethod
    def from_json(cls, config_file: str) -> "RouteConfig":
        with open(config_file, "r", encoding="utf-8") as f:
            c = json.load(f)

        distances = {item[0]: item[1] for item in c["distance"]}
        distances_list = [float(item[1]) for item in c["distance"]]

        road_loads = {hour: load for hour, load in c["road_loads"]}

        stop_ids = c.get("stop_ids", list(range(1, c["stop_number"] + 1)))

        raw_peak = c.get("peak_stop", stop_ids[len(stop_ids) // 2])
        peak_stop_index = (
            stop_ids.index(raw_peak) + 1 if raw_peak in stop_ids
            else len(stop_ids) // 2
        )

        config_route_id = str(c.get("route_id", config_file))
        
        target_intervals = None
        route_base = config_route_id.split("_")[0]
        try:
            with open(f"target_intervals/time_data_{route_base}.json", "r", encoding="utf-8") as f:
                time_data = json.load(f)
                target_intervals = time_data.get("target_intervals_minutes")
        except FileNotFoundError:
            pass  # It's fine if explicit route files are not present
        except Exception as e:
            log.warning(f"Could not load target intervals for route {route_base}: {e}")

        return cls(
            route_id=config_route_id,
            stop_ids=stop_ids,
            tram_capacity=c["tram_capacity"],
            flow_speed=c["flow_speed"],
            peak_stop_index=peak_stop_index,
            simulation_hours=c["simulation_hours"],
            distances=distances,
            road_loads=road_loads,
            distances_list=distances_list,
            depot_to_first_stop=c.get("depot_to_first_stop", 8.0),
            min_rest_time=c.get("min_rest_time", 15.0),
            turnaround_time=c.get("turnaround_time", DEFAULT_TURNAROUND),
            acceleration_time=c.get("acceleration_time", 0.5),
            stop_time=c.get("stop_time", 1.0),
            target_utilization=c.get("target_utilization", DEFAULT_TARGET_UTIL),
            random_seed=c.get("random_seed", None),
            target_intervals=target_intervals,
            contract_revenue_per_km=c.get("contract_revenue_per_km", 529.5),
            energy_per_km=c.get("energy_per_km", 0.0),
            maintenance_per_km=c.get("maintenance_per_km", 0.0),
            depreciation_per_km=c.get("depreciation_per_km", 0.0),
            payroll_per_trip=c.get("payroll_per_trip", 0.0),
            maintenance_fixed_per_trip=c.get("maintenance_fixed_per_trip", 0.0),
        )


class Route:
    """
    Один прогон маршрута (только fwd ИЛИ только bwd).
    Не знает о существовании парного маршрута — этим управляет MultiRoute.

    available_trams — откуда брать трамвай перед рейсом
    done_store      — куда класть трамвай после рейса
    """

    def __init__(
        self,
        config: RouteConfig,
        env: simpy.Environment,
        shared_stops: Dict[int, Stop],
        available_trams: simpy.Store,
        done_store: simpy.Store,
    ):
        self.config          = config
        self.env             = env
        self.shared_stops    = shared_stops
        self.available_trams = available_trams
        self.done_store      = done_store
        self.stats           = RouteStats(route_id=config.route_id)

    def start(self):
        self.env.process(self._continuous_dispatcher())

    # ── Вспомогательные ───────────────────────────────────────────────────────

    def _get_road_load(self, t_min: float) -> float:
        hour = (t_min // 60) % 24
        rl   = self.config.road_loads
        if not rl:
            return DEFAULT_ROAD_LOAD
        hours = sorted(rl)
        if hour in rl:
            return rl[hour]
        prev = [h for h in hours if h <= hour]
        nxt  = [h for h in hours if h > hour]
        if not prev:
            return rl[hours[0]]
        if not nxt:
            return rl[hours[-1]]
        h0, h1 = prev[-1], nxt[0]
        t = (hour - h0) / (h1 - h0)
        return rl[h0] * (1 - t) + rl[h1] * t

    def _calculate_travel_time(self, distance: float, t_min: float) -> float:
        if distance <= 0:
            return 0.0
        load  = self._get_road_load(t_min)
        speed = self.config.flow_speed * (1.0 - load)
        speed *= random.uniform(1.0 - SPEED_VARIATION, 1.0 + SPEED_VARIATION)
        speed = max(speed, MIN_SPEED_KMH)
        base_time = (distance / 1000.0) * (60.0 / speed) + self.config.acceleration_time
        
        traffic_light_delay = 0.0
        if random.random() < 0.30:
            traffic_light_delay = random.expovariate(1.0 / 0.7)
            
        return base_time + traffic_light_delay

    # ── SimPy-процессы ────────────────────────────────────────────────────────

    def _continuous_dispatcher(self):
        """
        Непрерывно выпускает трамваи по мере их появления в available_trams.
        Трамваи циклически проходят маршрут: fwd → разворот → bwd → отдых → fwd.
        """
        trip_id = 0
        sim_end = self.config.simulation_hours * 60
        while self.env.now < sim_end:
            tram = yield self.available_trams.get()
            trip_id += 1
            log.info(
                f"[{self.env.now:.1f}] Маршрут {self.config.route_id}: "
                f"трамвай #{tram.tram_id} выехал (рейс #{trip_id})"
            )
            self.env.process(self._tram_run(tram, trip_id))

    def _tram_run(self, tram: Tram, trip_id: int):
        """
        Один прогон трамвая по маршруту в одну сторону.
        После финальной остановки кладёт трамвай в done_store.

        Экономика: после завершения прогона рассчитываем доход и
        расчётных пассажиров на основе пройденного пробега и нормативов.
        """
        try:
            cfg = self.config
            tram.stats.total_trips += 1
            self.stats.total_trips += 1
            tram.direction = "forward" if "fwd" in cfg.route_id else "backward"

            trip_km = 0.0  # километраж за данный рейс

            for i, stop_id in enumerate(cfg.stop_ids):
                if i > 0:
                    distance    = cfg.distances_list[i - 1] if i - 1 < len(cfg.distances_list) else 0.0
                    travel_time = self._calculate_travel_time(distance, self.env.now)

                    km = distance / 1000.0
                    trip_km                       += km
                    self.stats.total_tram_km      += km

                    yield self.env.timeout(travel_time)

                yield self.env.process(
                    self._arrive_at_stop(tram, i + 1, stop_id, trip_id)
                )

            # ── Макро-экономическая оценка рейса ──────────────────────────────
            trip_passenger_revenue = trip_km * cfg.revenue_per_km
            trip_contract_revenue  = trip_km * cfg.contract_revenue_per_km
            trip_revenue        = trip_passenger_revenue + trip_contract_revenue
            trip_passengers_est = trip_km * cfg.passengers_per_km

            self.stats.total_passenger_revenue    += trip_passenger_revenue
            self.stats.total_contract_revenue     += trip_contract_revenue
            self.stats.total_revenue              += trip_revenue
            self.stats.total_passengers_estimated  += trip_passengers_est
            self.stats.total_passengers_served     += int(trip_passengers_est)

            tram.stats.passengers_served += int(trip_passengers_est)

            log.info(
                f"[{self.env.now:.1f}] Маршрут {cfg.route_id}: "
                f"трамвай #{tram.tram_id} завершил прогон "
                f"(рейс #{trip_id}, "
                f"km={trip_km:.2f}, "
                f"rev={trip_revenue:.0f} руб. (пасс: {trip_passenger_revenue:.0f}, контр: {trip_contract_revenue:.0f}), "
                f"pax_est={trip_passengers_est:.0f})"
            )

            yield self.done_store.put(tram)

        except simpy.Interrupt:
            log.warning(
                f"Трамвай #{tram.tram_id} "
                f"(маршрут {self.config.route_id}) прерван"
            )

    def _arrive_at_stop(
        self,
        tram: Tram,
        stop_index: int,
        stop_id: int,
        trip_id: int,
    ):
        """
        Обработка прибытия на остановку.

        Фиксированное время стоянки + расчёт headway error.
        """
        stop = self.shared_stops[stop_id]

        # Фиксированное время стоянки (≈1 мин)
        dwell_time = DEFAULT_STOP_DWELL_MIN
        departure_time = self.env.now + dwell_time

        stop.last_tram_time = self.env.now

        # ── Расчет headway error ──────────────────────────────────────────
        dep_hour = int(departure_time // 60) % 24
        target_headway = 0.0
        if self.config.target_intervals:
            keys = list(self.config.target_intervals.keys())
            if len(keys) >= 2:
                direction_key = keys[0] if tram.direction == "forward" else keys[1]
                if direction_key in self.config.target_intervals:
                    if dep_hour < 7:
                        hour_str = "before_07:00"
                    elif dep_hour == 23:
                        hour_str = "23:00-24:00"
                    else:
                        hour_str = f"{dep_hour:02d}:00-{dep_hour+1:02d}:00"
                    target_headway = self.config.target_intervals[direction_key].get(hour_str, 0.0)

        headway_error = 0.0
        route_id = self.config.route_id
        prev_departure = stop.last_tram_departure_time.get(route_id)
        if prev_departure is not None and target_headway > 0:
            actual_headway = departure_time - prev_departure
            headway_error = abs(actual_headway - target_headway)

        stop.last_tram_departure_time[route_id] = departure_time

        # ── Логирование ───────────────────────────────────────────────────
        tram.log_stop_event(
            time=self.env.now,
            stop_id=stop_id,
            direction=tram.direction,
            waiting_before=0,
            alighted=0,
            boarded=0,
            utilization_after=0.0,
            trip_id=trip_id,
            planned_time=None,
            headway_error=headway_error,
        )

        tram.log_schedule_deviation(
            stop_id=stop_id,
            planned_time=self.env.now,
            actual_time=self.env.now,
            headway_error=headway_error,
            route_id=self.config.route_id,
        )

        stop.log_event(StopEvent(
            time=self.env.now,
            route_id=self.config.route_id,
            tram_id=tram.tram_id,
            direction=tram.direction,
            waiting_before=0,
            alighted=0,
            boarded=0,
            passengers_in_tram=0,
            utilization_after=0.0,
        ))

        yield self.env.timeout(dwell_time)
