"""
Модели трамвая и его статистики. Не зависят от env и конфига маршрута.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List


@dataclass
class TramStats:
    tram_id: int
    route_id: str
    passengers_served: int = 0
    total_trips: int = 0
    utilization_history: List[float] = field(default_factory=list)
    stop_log: List[dict] = field(default_factory=list)
    schedule_deviations: List[dict] = field(default_factory=list)


class Tram:

    def __init__(self, tram_id: int, route_id: str, lightweight_mode: bool = False):
        self.tram_id = tram_id
        self.route_id = route_id
        self.passengers: int = 0
        self.direction: str = "forward"
        self.stats = TramStats(tram_id=tram_id, route_id=route_id)
        self.lightweight_mode = lightweight_mode

    @property
    def utilization(self) -> float:
        return 0.0

    def log_stop_event(
        self,
        time: float,
        stop_id: int,
        direction: str,
        waiting_before: int,
        alighted: int,
        boarded: int,
        utilization_after: float,
        trip_id: int = 0,
        planned_time: float = None,
        headway_error: float = None,
    ):
        if self.lightweight_mode:
            return
            
        self.stats.stop_log.append({
            "time":               time,
            "route_id":           self.route_id,
            "trip_id":            trip_id,
            "stop_id":            stop_id,
            "direction":          direction,
            "planned_time":       planned_time,
            "headway_error_min":  headway_error,
            "waiting_before":     waiting_before,
            "alighted":           alighted,
            "boarded":            boarded,
            "passengers_in_tram": self.passengers,
            "utilization_after":  utilization_after,
        })

    def log_schedule_deviation(
        self,
        stop_id: int,
        planned_time: float,
        actual_time: float,
        headway_error: float,
        route_id: str | None = None,
    ):
        self.stats.schedule_deviations.append({
            "tram_id":      self.tram_id,
            "route_id":     route_id or self.route_id,
            "stop_id":      stop_id,
            "planned_time": planned_time,
            "actual_time":  actual_time,
            "headway_error_min": headway_error,
        })
