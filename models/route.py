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
from typing import Dict, List, Optional, Tuple

import simpy

from models.stop import Stop, StopEvent
from models.tram import Tram
from constants import BOARDING_MIN_PER_PAX

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
    utilization_deviations: List[float] = field(default_factory=list)


@dataclass
class TripSchedule:
    """Эталонное расписание одного рейса."""
    trip_id: int
    departure_from_depot: float        # минуты от полуночи
    stop_times: Dict[int, float]       # {stop_id: planned_arrival_min}


@dataclass
class RouteConfig:
    route_id: str
    stop_ids: List[int]
    tram_capacity: int
    flow_speed: float
    peak_stop_index: int
    simulation_hours: int
    distances: Dict[int, float]
    intensity_map: Dict[int, Dict[int, float]]
    schedule: List[TripSchedule]
    road_loads: Dict[int, float]
    depot_to_first_stop: float = 8.0
    min_rest_time: float = 15.0
    turnaround_time: float = DEFAULT_TURNAROUND
    acceleration_time: float = 0.5
    stop_time: float = 1.0
    target_utilization: float = DEFAULT_TARGET_UTIL
    random_seed: Optional[int] = None
    target_intervals: Optional[Dict[str, Dict[str, int]]] = None

    @property
    def stop_number(self) -> int:
        return len(self.stop_ids)

    @classmethod
    def from_json(cls, config_file: str) -> "RouteConfig":
        with open(config_file, "r", encoding="utf-8") as f:
            c = json.load(f)

        distances = {item[0]: item[1] for item in c["distance"]}

        intensity_map: Dict[int, Dict[int, float]] = defaultdict(dict)
        for stop_id, hour, intensity in c["intensity"]:
            intensity_map[stop_id][hour] = intensity

        road_loads = {hour: load for hour, load in c["road_loads"]}

        stop_ids = c.get("stop_ids", list(range(1, c["stop_number"] + 1)))

        raw_peak = c.get("peak_stop", stop_ids[len(stop_ids) // 2])
        peak_stop_index = (
            stop_ids.index(raw_peak) + 1 if raw_peak in stop_ids
            else len(stop_ids) // 2
        )

        schedule: List[TripSchedule] = []
        for trip in c.get("schedule", []):
            stop_times = {stop_id: arr_min for stop_id, arr_min in trip["stops"]}
            schedule.append(TripSchedule(
                trip_id=trip["trip_id"],
                departure_from_depot=trip["departure_from_depot"],
                stop_times=stop_times,
            ))

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
            intensity_map=dict(intensity_map),
            schedule=schedule,
            road_loads=road_loads,
            depot_to_first_stop=c.get("depot_to_first_stop", 8.0),
            min_rest_time=c.get("min_rest_time", 15.0),
            turnaround_time=c.get("turnaround_time", DEFAULT_TURNAROUND),
            acceleration_time=c.get("acceleration_time", 0.5),
            stop_time=c.get("stop_time", 1.0),
            target_utilization=c.get("target_utilization", DEFAULT_TARGET_UTIL),
            random_seed=c.get("random_seed", None),
            target_intervals=target_intervals,
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
        self.env.process(self._schedule_dispatcher())

    # ── Вспомогательные ───────────────────────────────────────────────────────

    def _get_intensity(self, stop_id: int, hour: int) -> float:
        return self.config.intensity_map.get(stop_id, {}).get(hour, 0.0)

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

    def _schedule_dispatcher(self):
        """
        Читает schedule и выпускает трамваи строго по расписанию.
        Если трамвая нет в момент отправления — рейс пропускается.
        """
        for trip in self.config.schedule:
            # Ждём планового времени выезда из депо
            wait = trip.departure_from_depot - self.env.now
            if wait > 0:
                yield self.env.timeout(wait)

            # Трамвай есть — берём, нет — пропускаем рейс
            if len(self.available_trams.items) > 0:
                tram = yield self.available_trams.get()
                departure_delay = self.env.now - trip.departure_from_depot
                log.info(
                    f"[{self.env.now:.1f}] Маршрут {self.config.route_id}: "
                    f"трамвай #{tram.tram_id} выехал "
                    f"(рейс #{trip.trip_id}, "
                    f"план={trip.departure_from_depot:.1f}, "
                    f"задержка={departure_delay:+.1f} мин)"
                )
                self.env.process(self._tram_run(tram, trip))
            else:
                log.warning(
                    f"[{self.env.now:.1f}] Маршрут {self.config.route_id}: "
                    f"рейс #{trip.trip_id} ПРОПУЩЕН — нет свободных трамваев "
                    f"(план={trip.departure_from_depot:.1f})"
                )

    def _tram_run(self, tram: Tram, trip: TripSchedule):
        """
        Один прогон трамвая по маршруту в одну сторону.
        После финальной остановки кладёт трамвай в done_store.
        """
        try:
            cfg = self.config
            tram.stats.total_trips += 1
            tram.direction = "forward" if "fwd" in cfg.route_id else "backward"

            for i, stop_id in enumerate(cfg.stop_ids):
                if i > 0:
                    distance    = cfg.distances.get(stop_id, 0.0)
                    travel_time = self._calculate_travel_time(distance, self.env.now)

                    km = distance / 1000.0
                    self.stats.total_tram_km      += km
                    self.stats.total_passenger_km += km * tram.passengers

                    yield self.env.timeout(travel_time)

                yield self.env.process(
                    self._arrive_at_stop(tram, i + 1, stop_id, trip)
                )

            log.info(
                f"[{self.env.now:.1f}] Маршрут {cfg.route_id}: "
                f"трамвай #{tram.tram_id} завершил прогон "
                f"(рейс #{trip.trip_id})"
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
        trip: TripSchedule,
    ):
        stop            = self.shared_stops[stop_id]
        hour            = int(self.env.now // 60) % 24
        time_since_last = self.env.now - stop.last_tram_time
        waiting_before  = stop.waiting_passengers
        planned         = trip.stop_times.get(stop_id)

        alighted = tram.alight_passengers(
            stop_index, self.config.stop_number, self.config.peak_stop_index
        )

        new_pax = stop.get_new_passengers(
            self._get_intensity(stop_id, hour), time_since_last
        )
        stop.waiting_passengers += new_pax
        stop.record_waiting()

        boarded = tram.board_passengers(stop.waiting_passengers)
        stop.waiting_passengers -= boarded
        stop.record_waiting()

        if boarded > 0:
            stop.add_waiting_time(boarded, time_since_last)
        stop.last_tram_time = self.env.now

        tram.stats.passengers_served       += boarded
        self.stats.total_passengers_served += boarded
        if not tram.lightweight_mode:
            tram.stats.utilization_history.append(tram.utilization)
            self.stats.utilization_deviations.append(
                abs(tram.utilization - self.config.target_utilization)
            )

        effective_pax_flow = max(boarded, alighted)
        boarding_time = effective_pax_flow * BOARDING_MIN_PER_PAX
        departure_time = self.env.now + self.config.stop_time + boarding_time

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
        if stop.last_tram_departure_time is not None and target_headway > 0:
            actual_headway = departure_time - stop.last_tram_departure_time
            headway_error = abs(actual_headway - target_headway)
            
        stop.last_tram_departure_time = departure_time

        # Единственное место где логируем — с trip_id и planned_time
        tram.log_stop_event(
            time=self.env.now,
            stop_id=stop_id,
            direction=tram.direction,
            waiting_before=waiting_before + new_pax,
            alighted=alighted,
            boarded=boarded,
            utilization_after=tram.utilization * 100,
            trip_id=trip.trip_id,
            planned_time=planned,
            headway_error=headway_error,
        )

        # Отклонение от расписания
        if planned is not None:
            tram.log_schedule_deviation(
                stop_id=stop_id,
                planned_time=planned,
                actual_time=self.env.now,
                headway_error=headway_error,
            )

        stop.log_event(StopEvent(
            time=self.env.now,
            route_id=self.config.route_id,
            tram_id=tram.tram_id,
            direction=tram.direction,
            waiting_before=waiting_before + new_pax,
            alighted=alighted,
            boarded=boarded,
            passengers_in_tram=tram.passengers,
            utilization_after=tram.utilization,
        ))

        yield self.env.timeout(self.config.stop_time + boarding_time)
