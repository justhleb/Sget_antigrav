"""
Модель остановки. Остановка живёт в глобальном реестре MultiRouteSimulation
и может обслуживать трамваи любого маршрута.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import List

import simpy
import numpy as np


@dataclass
class StopEvent:
    """Одно событие прибытия трамвая на остановку (для логирования)."""
    time: float
    route_id: str
    tram_id: int
    direction: str
    waiting_before: int
    alighted: int
    boarded: int
    passengers_in_tram: int
    utilization_after: float   # 0..1


class Stop:
    """Остановка в общем реестре симуляции."""

    def __init__(self, stop_id: int, env: simpy.Environment):
        self.stop_id = stop_id
        self.env = env

        self.waiting_passengers: int = 0
        self.last_tram_time: float = 0.0
        self.last_tram_departure_time: float = None

        # Агрегированная статистика
        self.total_waiting_time: float = 0.0
        self.passengers_served: int = 0

        # История для визуализации
        self.waiting_history: List[tuple] = []   # (time, count)
        self.event_log: List[StopEvent] = []     # детальные события
        self.lightweight_mode: bool = False

    def record_waiting(self):
        if not self.lightweight_mode:
            self.waiting_history.append((self.env.now, self.waiting_passengers))

    def add_waiting_time(self, boarded: int, time_since_last: float):
        """Среднее время ожидания = половина интервала (равномерное прибытие)."""
        self.total_waiting_time += boarded * (time_since_last / 2.0)
        self.passengers_served += boarded

    def get_new_passengers(self, intensity_per_hour: float, time_since_last: float) -> int:
        """Новые пассажиры за время ожидания (Пуассоновское распределение)."""
        if time_since_last <= 0 or intensity_per_hour <= 0:
            return 0
        rate = intensity_per_hour * (time_since_last / 60.0)
        return int(np.random.poisson(rate)) if rate > 0 else 0

    def log_event(self, event: StopEvent):
        if not self.lightweight_mode:
            self.event_log.append(event)

    @property
    def avg_waiting_time(self) -> float:
        if self.passengers_served == 0:
            return 0.0
        return self.total_waiting_time / self.passengers_served
