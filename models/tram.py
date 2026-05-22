"""
Модели трамвая и его статистики.

Содержит описание подвижного состава (Трамвай), логику сбора его индивидуальной
эксплуатационной статистики и методы логирования событий прибытий на остановки.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List


@dataclass
class TramStats:
    """
    Класс для сбора и агрегации индивидуальной статистики работы трамвайного вагона.
    """
    tram_id: int              # Уникальный ID трамвая
    route_id: str             # ID текущего маршрута (например, "20_fwd")
    passengers_served: int = 0  # Общее число перевезенных пассажиров (устарело, сохранено для совместимости)
    total_trips: int = 0      # Общее число выполненных рейсов (полурейсов) за операционный день
    utilization_history: List[float] = field(default_factory=list) # История коэффициента загрузки вагона (устарело)
    
    # Лог остановок. Хранит подробную информацию о каждом прибытии вагона на остановку.
    # Используется логгером для выгрузки индивидуальных CSV-логов трамваев.
    stop_log: List[dict] = field(default_factory=list)
    
    # Лог отклонений от расписания (headway error). Хранит информацию о запланированном
    # и фактическом времени прибытия на каждую остановку, а также величину ошибки интервала.
    schedule_deviations: List[dict] = field(default_factory=list)


class Tram:
    """
    Представляет трамвайный вагон в симуляционной модели.
    
    Отвечает за сохранение своего текущего состояния движения (направление, идентификаторы),
    а также за агрегацию и логирование событий в процессе выполнения рейсов.
    """

    def __init__(self, tram_id: int, route_id: str, lightweight_mode: bool = False):
        """
        Инициализация вагона.

        :param tram_id: Уникальный номер трамвая
        :param route_id: Базовый номер маршрута (например, "20")
        :param lightweight_mode: Флаг легкого режима (отключает ведение детальных логов)
        """
        self.tram_id = tram_id
        self.route_id = route_id
        self.passengers: int = 0   # Количество пассажиров в вагоне (сейчас всегда 0, так как экономика макро-уровня)
        self.direction: str = "forward"  # Текущее направление движения: "forward" или "backward"
        self.stats = TramStats(tram_id=tram_id, route_id=route_id)
        self.lightweight_mode = lightweight_mode

    @property
    def utilization(self) -> float:
        """
        Текущий коэффициент загрузки вагона.
        Всегда равен 0.0, так как расчеты перенесены на макро-экономический уровень (Top-Down).
        """
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
        """
        Записывает событие прохождения остановки трамваем в историю стоп-лога вагона.

        :param time: Время симуляции в минутах
        :param stop_id: Уникальный ID остановки
        :param direction: Направление ("forward" / "backward")
        :param waiting_before: Кол-во ждущих пассажиров (0)
        :param alighted: Кол-во высадившихся (0)
        :param boarded: Кол-во севших (0)
        :param utilization_after: Загрузка вагона (0.0)
        :param trip_id: Порядковый номер рейса вагона за день
        :param planned_time: Запланированное время прибытия
        :param headway_error: Отклонение от целевого интервала (в минутах)
        """
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
        """
        Записывает отклонение трамвая от планового интервала движения (headway error) на остановке.
        Используется в дальнейшем для расчета MAE маршрута и записи в schedule_deviations.csv.

        :param stop_id: ID остановки
        :param planned_time: Плановое время отправления/выпуска
        :param actual_time: Фактическое время отправления/выпуска
        :param headway_error: Величина отклонения в минутах
        :param route_id: Конкретное направление маршрута (например, "20_fwd")
        """
        self.stats.schedule_deviations.append({
            "tram_id":      self.tram_id,
            "route_id":     route_id or self.route_id,
            "stop_id":      stop_id,
            "planned_time": planned_time,
            "actual_time":  actual_time,
            "headway_error_min": headway_error,
        })

