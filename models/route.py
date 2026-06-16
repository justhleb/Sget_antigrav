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

from models.stop import Stop
from models.tram import Tram
from constants import (
    DEFAULT_STOP_DWELL_MIN,
    SPEED_VARIATION,
    DEFAULT_TURNAROUND,
)

log = logging.getLogger(__name__)


@dataclass
class RouteStats:
    """
    Класс для накопления агрегированной статистики по конкретному направлению маршрута
    (например, отдельно для 20_fwd и 20_bwd) за весь операционный день.
    """
    route_id: str
    total_passengers_served: int = 0      # Общее число обслуженных пассажиров (целое число)
    total_tram_km: float = 0.0            # Общий пробег всех трамваев на этом маршруте (км)
    total_passenger_km: float = 0.0       # Общий пассажиро-километраж (устарело)
    total_passenger_revenue: float = 0.0  # руб. Суммарный доход от пассажиров (пробег * pax_per_km * fare)
    total_contract_revenue: float = 0.0   # руб. Суммарный полученный доход от исполнения транспортного контракта
    total_revenue: float = 0.0           # руб. Совокупный доход (пассажирский + контрактный)
    total_passengers_estimated: float = 0.0 # Расчётное (вещественное) число перевезенных лиц
    total_trips: int = 0                 # Общее число выполненных рейсов (полурейсов) за день
    utilization_deviations: List[float] = field(default_factory=list) # Отклонения загрузки (устарело)
    failed_release_trips: int = 0         # Количество рейсов с нарушением интервала выпуска из депо (Type 1)
    failed_midpoint_trips: int = 0        # Количество рейсов с нарушением интервала на серединной остановке (Type 2)
    lost_contract_revenue: float = 0.0    # руб. Неполученный (упущенный) доход от контракта из-за штрафов Type 1




@dataclass
class RouteConfig:
    """
    Конфигурация одного направления маршрута.
    Считывается из соответствующего JSON-файла в папке configs/.
    """
    route_id: str                  # Идентификатор направления (например, "20_fwd")
    stop_ids: List[int]            # Список уникальных ID остановок в порядке их следования
    flow_speed: float              # Базовая скорость движения (устарело, переопределяется константой 17 км/ч в коде)
    peak_stop_index: int           # Индекс серединной остановки (для контроля отклонений Type 2)
    simulation_hours: int          # Длительность симуляции в часах (обычно 24)
    distances: Dict[int, float]    # Расстояния до остановок от начала маршрута (в метрах)
    road_loads: Dict[int, float]   # Коэффициенты загрузки дорог по часам (fallback, если нет road_load.json)
    distances_list: List[float] = field(default_factory=list) # Массив расстояний между соседними остановками (в метрах)
    depot_to_first_stop: float = 0.0 # Время выезда из депо до 1-й остановки (в минутах)
    min_rest_time: float = 3.0      # Минимальное время отдыха водителя на конечной станции (в минутах)
    turnaround_time: float = DEFAULT_TURNAROUND # Время разворота на конечном кольце (в минутах)
    acceleration_time: float = 10.0 / 60.0 # Время разгона и торможения на каждой остановке (в минутах)
    stop_time: float = 1.0          # Время стоянки на остановке по умолчанию (устарело, dwell_time фиксирован на 0.5 мин)
    random_seed: Optional[int] = None # Сид для воспроизводимости случайных величин
    target_intervals: Optional[Dict[str, Dict[str, int]]] = None # Целевые интервалы по часам из target_intervals/
    penalties_config: dict = field(default_factory=dict) # Настройки штрафов и допусков из penalties_config.json
    
    # ── Экономические нормативы (считываются из JSON) ──
    revenue_per_km: float = 0.0       # руб. Средний исторический доход от пассажиров на 1 км пути
    passengers_per_km: float = 0.0    # чел. Среднее историческое число пассажиров на 1 км пути
    
    # ── Расходные и доходные параметры контракта (считываются из JSON) ──
    contract_revenue_per_km: float = 529.5 # руб/км. Тариф оплаты транспортной работы по брутто-контракту
    energy_per_trip: float = 0.0            # руб/рейс. Расход электроэнергии на один рейс
    depreciation_per_trip: float = 0.0      # руб/рейс. Амортизационные отчисления за рейс
    payroll_per_trip: float = 0.0            # руб/рейс. ФОТ водителя за один выполненный рейс (полурейс)
    maintenance_per_trip: float = 0.0        # руб/рейс. Платеж за ТОиР вагона за рейс
    tram_count: Optional[int] = None        # Требуемое контрактное количество трамваев на маршруте

    @property
    def stop_number(self) -> int:
        return len(self.stop_ids)

    @classmethod
    def from_json(cls, config_file: str) -> "RouteConfig":
        """
        Загружает конфигурацию направления маршрута из JSON-файла.
        Автоматически подгружает почасовые коэффициенты дорожной загрузки 
        из внешнего файла configs/road_load.json.
        
        :param config_file: Путь к файлу конфигурации JSON.
        :return: Экземпляр RouteConfig с заполненными параметрами.
        """
        with open(config_file, "r", encoding="utf-8") as f:
            c = json.load(f)

        distances = {item[0]: item[1] for item in c["distance"]}
        distances_list = [float(item[1]) for item in c["distance"]]

        # Загружаем коэффициенты из внешнего файла configs/road_load.json
        try:
            with open("configs/road_load.json", "r", encoding="utf-8") as rf:
                rl_data = json.load(rf)
            road_loads = {int(hour): float(load) for hour, load in rl_data["road_loads"]}
        except Exception as e:
            log.warning(f"Failed to load road_load.json, fallback to config road_loads: {e}")
            road_loads = {hour: load for hour, load in c["road_loads"]}

        stop_ids = c.get("stop_ids", list(range(1, c["stop_number"] + 1)))

        # Загружаем целевые интервалы движения из папки target_intervals/
        target_intervals = None
        route_base = c["route_id"].split("_")[0]
        try:
            with open(f"target_intervals/time_data_{route_base}.json", "r", encoding="utf-8") as tf:
                time_data = json.load(tf)
                target_intervals = time_data.get("target_intervals_minutes")
        except FileNotFoundError:
            pass  # Допускается отсутствие файлов интервалов для некоторых тестов
        except Exception as e:
            log.warning(f"Не удалось загрузить целевые интервалы для маршрута {route_base}: {e}")

        return cls(
            route_id=c["route_id"],
            stop_ids=stop_ids,
            flow_speed=c.get("flow_speed", 17.0),
            peak_stop_index=c.get("peak_stop_index", len(stop_ids) // 2),
            simulation_hours=c.get("simulation_hours", 24),
            distances=distances,
            road_loads=road_loads,
            distances_list=distances_list,
            depot_to_first_stop=c.get("depot_to_first_stop", 0.0),
            min_rest_time=c.get("min_rest_time", 3.0),
            turnaround_time=c.get("turnaround_time", DEFAULT_TURNAROUND),
            acceleration_time=c.get("acceleration_time", 10.0 / 60.0),
            stop_time=c.get("stop_time", 1.0),
            random_seed=c.get("random_seed"),
            target_intervals=target_intervals,
            penalties_config=c.get("penalties_config", {}),
            contract_revenue_per_km=c.get("contract_revenue_per_km", 529.5),
            energy_per_trip=c.get("energy_per_trip", 0.0),
            depreciation_per_trip=c.get("depreciation_per_trip", 0.0),
            payroll_per_trip=c.get("payroll_per_trip", 0.0),
            maintenance_per_trip=c.get("maintenance_per_trip", 0.0),
            tram_count=c.get("tram_count"),
            revenue_per_km=c.get("revenue_per_km", 0.0),
            passengers_per_km=c.get("passengers_per_km", 0.0),
        )
class Route:
    """
    Класс, представляющий одно направление движения трамвайного маршрута 
    (например, только "туда" (fwd) или только "обратно" (bwd)).
    
    Он моделирует движение трамваев по остановкам, рассчитывает время в пути с учетом 
    загрузки дорог, фиксирует проезд остановок и собирает экономическую/эксплуатационную 
    статистику за день.
    
    Класс автономен и не знает о существовании противоположного направления; связь 
    между ними и полный цикл (fwd -> разворот -> bwd -> отдых) координирует MultiRoute.
    """

    def __init__(
        self,
        config: RouteConfig,
        env: simpy.Environment,
        shared_stops: Dict[int, Stop],
        available_trams: simpy.Store,
        done_store: simpy.Store,
    ):
        """
        Инициализация объекта направления маршрута.

        :param config: Конфигурация данного направления (RouteConfig).
        :param env: Окружение симуляции SimPy.
        :param shared_stops: Словарь всех остановок в симуляции, доступных разным маршрутам.
        :param available_trams: Хранилище (SimPy Store) готовых к отправке трамваев на конечной станции.
        :param done_store: Хранилище (SimPy Store) для трамваев, успешно завершивших рейс.
        """
        self.config          = config
        self.env             = env
        self.shared_stops    = shared_stops
        self.available_trams = available_trams
        self.done_store      = done_store
        self.stats           = RouteStats(route_id=config.route_id)

    def start(self):
        """
        Запуск процесса диспетчеризации рейсов на данном направлении.
        """
        self.env.process(self._continuous_dispatcher())

    # ── Вспомогательные ───────────────────────────────────────────────────────

    def _get_road_load(self, t_min: float) -> float:
        """
        Определяет коэффициент загрузки дорог (road load coefficient) для текущего момента времени.
        
        Использует значение из словаря road_loads. Если точного совпадения по часу нет,
        выполняет линейную интерполяцию между соседними часами.
        
        :param t_min: Текущее модельное время в минутах от начала симуляции.
        :return: Коэффициент загрузки (например, 0.65 при снижении скорости на 35%).
        """
        hour = (t_min // 60) % 24
        rl   = self.config.road_loads
        if not rl:
            return 1.0
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
        """
        Рассчитывает время движения трамвая между остановками.
        
        Скорость движения вычисляется как: базовая_скорость * коэффициент_загрузки * случайная_флуктуация.
        К итоговому времени движения прибавляется время на разгон и торможение, а также 
        случайная задержка на светофорах с вероятностью 10%.
        
        :param distance: Расстояние между остановками в метрах.
        :param t_min: Текущее модельное время в минутах.
        :return: Итоговое время движения в минутах.
        """
        if distance <= 0:
            return 0.0
        load  = self._get_road_load(t_min)
        speed = self.config.flow_speed * load
        speed *= random.uniform(1.0 - SPEED_VARIATION, 1.0 + SPEED_VARIATION)
        base_time = (distance / 1000.0) * (60.0 / speed) + self.config.acceleration_time
        
        traffic_light_delay = 0.0
        if random.random() < 0.10:
            traffic_light_delay = random.expovariate(1.0 / (20.0 / 60.0))
            
        return base_time + traffic_light_delay

    # ── SimPy-процессы ────────────────────────────────────────────────────────

    def _continuous_dispatcher(self):
        """
        SimPy-процесс: непрерывный выпуск трамваев на маршрут по мере их готовности.
        
        Извлекает доступные трамваи из available_trams и запускает для каждого
        из них отдельный асинхронный процесс прогона по маршруту (_tram_run).
        """
        from constants import LAST_TRIP_DEPARTURE_MIN
        
        trip_id = 0
        is_fwd = "fwd" in self.config.route_id
        
        # Для fwd ограничиваем выпуск до 23:30 (LAST_TRIP_DEPARTURE_MIN).
        # Для bwd разрешаем движение до конца работы (30 часов), чтобы все
        # запущенные fwd-рейсы могли благополучно вернуться в депо.
        limit_time = LAST_TRIP_DEPARTURE_MIN if is_fwd else 30 * 60
        
        while self.env.now < limit_time:
            tram = yield self.available_trams.get()
            if self.env.now >= limit_time:
                yield self.available_trams.put(tram)
                break
            trip_id += 1
            log.info(
                f"[{self.env.now:.1f}] Маршрут {self.config.route_id}: "
                f"трамвай #{tram.tram_id} выехал (рейс #{trip_id})"
            )
            self.env.process(self._tram_run(tram, trip_id))

    def _tram_run(self, tram: Tram, trip_id: int):
        """
        SimPy-процесс: моделирование прогона конкретного трамвая по всему направлению маршрута.
        
        Проходит по всем остановкам последовательно, считает время перемещения,
        инициирует процесс остановки на каждой из них. После завершения прогона
        рассчитывает макроэкономические показатели за рейс (выручку, пассажиропоток),
        проверяет наличие нарушений расписания (Type 1, Type 2) и возвращает трамвай в done_store.
        
        :param tram: Объект проходящего по маршруту трамвая.
        :param trip_id: Уникальный ID текущего рейса в рамках этого направления.
        """
        try:
            cfg = self.config
            tram.stats.total_trips += 1
            self.stats.total_trips += 1
            tram.direction = "forward" if "fwd" in cfg.route_id else "backward"
            tram.current_trip_midpoint_failed = False

            # Проверка Type 1: Нарушение интервала выпуска (только для forward отправлений)
            release_failed = False
            is_fwd = "fwd" in cfg.route_id
            if is_fwd:
                target_rel = getattr(tram, "target_release_time", None)
                actual_rel = getattr(tram, "actual_release_time", None)
                if target_rel is not None and actual_rel is not None:
                    deviation = actual_rel - target_rel
                    early_tol = cfg.penalties_config.get("release_early_tolerance_min", 1.0)
                    late_tol = cfg.penalties_config.get("release_late_tolerance_min", 5.0)
                    if deviation < -early_tol or deviation > late_tol:
                        release_failed = True

            trip_km = 0.0  # Пробег за текущий рейс

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

            # ── Макро-экономическая оценка выполненного рейса ──────────────────
            trip_passenger_revenue = trip_km * cfg.revenue_per_km
            trip_contract_revenue  = trip_km * cfg.contract_revenue_per_km

            # Если выпуск из депо был сорван, контрактный доход за этот рейс обнуляется (штраф)
            if release_failed:
                self.stats.failed_release_trips += 1
                self.stats.lost_contract_revenue += trip_contract_revenue
                actual_contract_revenue = 0.0
            else:
                actual_contract_revenue = trip_contract_revenue

            trip_revenue        = trip_passenger_revenue + actual_contract_revenue
            trip_passengers_est = trip_km * cfg.passengers_per_km

            self.stats.total_passenger_revenue    += trip_passenger_revenue
            self.stats.total_contract_revenue     += actual_contract_revenue
            self.stats.total_revenue              += trip_revenue
            self.stats.total_passengers_estimated  += trip_passengers_est
            self.stats.total_passengers_served     += int(trip_passengers_est)

            tram.stats.passengers_served += int(trip_passengers_est)

            # Проверка Type 2: Было ли нарушение интервала на серединной остановке
            if getattr(tram, "current_trip_midpoint_failed", False):
                self.stats.failed_midpoint_trips += 1

            log.info(
                f"[{self.env.now:.1f}] Маршрут {cfg.route_id}: "
                f"трамвай #{tram.tram_id} завершил прогон "
                f"(рейс #{trip_id}, "
                f"km={trip_km:.2f}, "
                f"rev={trip_revenue:.0f} руб. (пасс: {trip_passenger_revenue:.0f}, контр: {trip_contract_revenue:.0f}), "
                f"pax_est={trip_passengers_est:.0f})"
            )

            # Передаем трамвай на конечную станцию (в MultiRoute для разворота или отдыха)
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
        SimPy-процесс: Обработка прибытия трамвая на остановку.
        
        Трамвай стоит на остановке фиксированное время dwell_time (0.5 минуты),
        а также вычисляются отклонения от целевого интервала движения (headway error).
        Если на серединной остановке отклонение превышает допустимый лимит (Type 2),
        рейс отмечается как неуспешный.
        События проезда логируются в статистику трамвая.
        
        :param tram: Объект трамвая.
        :param stop_index: Порядковый номер остановки на маршруте (1-indexed).
        :param stop_id: Уникальный ID остановки.
        :param trip_id: ID текущего рейса.
        """
        stop = self.shared_stops[stop_id]

        # Фиксированное время стоянки (dwell time)
        dwell_time = DEFAULT_STOP_DWELL_MIN
        departure_time = self.env.now + dwell_time

        stop.last_tram_time = self.env.now

        # ── Расчет отклонения от целевого интервала (headway error) ──────────
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

        # Проверка Type 2: Превышение интервала на серединной остановке
        is_midpoint = (stop_index - 1 == len(self.config.stop_ids) // 2)
        if is_midpoint and prev_departure is not None and target_headway > 0:
            mid_tol = self.config.penalties_config.get("midpoint_tolerance_min", 5.0)
            if headway_error > mid_tol:
                tram.current_trip_midpoint_failed = True

        stop.last_tram_departure_time[route_id] = departure_time

        # ── Запись событий в лог и статистику ──────────────────────────────────
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

        yield self.env.timeout(dwell_time)
