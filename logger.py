"""
Модуль логирования детальных событий трамваев.

Этот модуль отвечает за сбор, форматирование и сохранение в файлы формата CSV данных 
о движении трамваев, их проездах через остановки и отклонениях от запланированных интервалов.
"""
from __future__ import annotations

import csv
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any

log = logging.getLogger(__name__)


class TramLogger:
    """
    Класс для записи логов работы трамваев в CSV файлы.
    
    Позволяет сохранять детальные логи прохождения остановок каждым трамваем,
    а также сводный отчет по отклонениям от интервалов для всех трамваев.
    """

    # Колонки для детального лога проезда остановок конкретным трамваем
    TRAM_LOG_COLUMNS = [
        "tram_id",
        "trip_id",           # Уникальный ID рейса в рамках смены трамвая
        "time_min",          # Время события в минутах от начала симуляции
        "headway_error_min", # Величина отклонения от целевого интервала (в минутах)
        "hour",              # Час суток, в который произошло событие
        "stop_id",           # ID остановки, на которой зафиксировано событие
    ]

    # Колонки для единого сводного лога отклонений от расписания
    DEVIATION_COLUMNS = [   
        "tram_id",           # ID трамвая
        "stop_id",           # ID остановки
        "planned_time",      # Запланированное время прибытия по интервалу (мин)
        "actual_time",       # Фактическое время прибытия трамвая (мин)
        "headway_error_min", # Ошибка интервала (фактический интервал минус целевой)
    ]

    def __init__(
        self,
        output_dir: str = "tram_logs",
        file_prefix: str = "tram",
        write_header: bool = True,
    ):
        """
        Инициализирует логгер трамваев.

        :param output_dir: Путь к директории, в которую будут сохраняться CSV-файлы.
        :param file_prefix: Префикс названия файлов для индивидуальных логов трамваев.
        :param write_header: Флаг, определяющий, нужно ли записывать заголовки колонок в CSV.
        """
        self.output_dir   = Path(output_dir)
        self.file_prefix  = file_prefix
        self.write_header = write_header
        self._ensure_output_dir()

    def _ensure_output_dir(self) -> None:
        """Создает выходную директорию, если она еще не существует."""
        self.output_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _direction_label(direction: str) -> str:
        """Вспомогательный метод для нормализации направления движения."""
        return direction if direction in ("forward", "backward") else str(direction)

    @staticmethod
    def _safe_int(x: Any, default: int = 0) -> int:
        """Безопасное приведение значения к целому числу с дефолтным значением при ошибке."""
        try:    return int(x)
        except: return default

    @staticmethod
    def _safe_float(x: Any, default: float = 0.0) -> float:
        """Безопасное приведение значения к вещественному числу с дефолтным значением при ошибке."""
        try:    return float(x)
        except: return default

    # ── Лог остановок трамвая ─────────────────────────────────────────────────

    def save_tram_log(
        self,
        tram_id: int,
        stop_log: List[dict],
        route_id: Optional[str] = None,
    ) -> Optional[Path]:
        """
        Сохраняет детальный лог прохождения остановок конкретным трамваем в отдельный CSV-файл.

        :param tram_id: Уникальный идентификатор трамвая.
        :param stop_log: Список словарей с событиями проезда остановок.
        :param route_id: Необязательный ID маршрута трамвая.
        :return: Path к созданному файлу или None, если лог пуст.
        """
        if not stop_log:
            log.info(f"Tram #{tram_id}: no stop events to save")
            return None

        filename = f"{self.file_prefix}_{tram_id:03d}.csv"
        filepath = self.output_dir / filename

        with filepath.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=self.TRAM_LOG_COLUMNS)
            if self.write_header:
                writer.writeheader()

            for event in stop_log:
                t        = self._safe_float(event.get("time", 0.0))
                headway_error = event.get("headway_error_min")
                writer.writerow({
                    "tram_id":            tram_id,
                    "trip_id":            self._safe_int(event.get("trip_id", 0)),
                    "time_min":           round(t, 4),
                    "headway_error_min":  round(float(headway_error), 4) if headway_error is not None else "",
                    "hour":               int(t // 60) % 24,
                    "stop_id":            self._safe_int(event.get("stop_id", 0)),
                })

        log.info(f"Tram #{tram_id}: {len(stop_log)} stop events → {filepath.name}")
        return filepath

    def save_all_trams(
        self,
        trams: Dict[int, Any],
        route_id: Optional[str] = None,
        include_empty: bool = False,
    ) -> List[Path]:
        """
        Сохраняет индивидуальные логи прохождения остановок для всех переданных трамваев.

        :param trams: Словарь {tram_id: объект_трамвая}.
        :param route_id: Необязательный ID маршрута.
        :param include_empty: Записывать ли пустые файлы для трамваев без событий.
        :return: Список путей к сохраненным CSV-файлам.
        """
        log.info("=" * 60)
        log.info("SAVING TRAM LOGS")
        log.info("=" * 60)

        paths: List[Path] = []
        for tram_id, tram in trams.items():
            stop_log = getattr(getattr(tram, "stats", None), "stop_log", None) or []
            if stop_log:
                p = self.save_tram_log(tram_id, stop_log, route_id=route_id)
                if p is not None:
                    paths.append(p)
            elif include_empty:
                filename = f"{self.file_prefix}_{tram_id:03d}.csv"
                filepath = self.output_dir / filename
                with filepath.open("w", newline="", encoding="utf-8") as f:
                    writer = csv.DictWriter(f, fieldnames=self.TRAM_LOG_COLUMNS)
                    if self.write_header:
                        writer.writeheader()
                paths.append(filepath)

        log.info("-" * 60)
        log.info(f"Saved logs: {len(paths)} / {len(trams)}")
        log.info(f"Folder: {self.output_dir.resolve()}")
        log.info("=" * 60)
        return paths

    # ── Отклонения от расписания (отдельный CSV) ──────────────────────────────

    def save_schedule_deviations(
        self,
        trams: Dict[int, Any],
        output_file: str = "schedule_deviations.csv",
        route_id: Optional[str] = None,
    ) -> Path:
        """
        Сохраняет все зафиксированные отклонения от расписания (целевых интервалов)
        по всем трамваям в один сводный CSV-файл.
        
        Этот файл очень удобен для последующего анализа качества движения 
        (например, с помощью pandas) и построения распределения ошибок интервалов.

        :param trams: Словарь {tram_id: объект_трамвая}.
        :param output_file: Имя результирующего CSV-файла.
        :param route_id: Необязательный ID маршрута.
        :return: Путь к созданному файлу отклонений.
        """
        filepath = self.output_dir / output_file
        total    = 0

        with filepath.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=self.DEVIATION_COLUMNS)
            writer.writeheader()

            for tram_id, tram in sorted(trams.items(), key=lambda x: x[0]):
                devs = getattr(getattr(tram, "stats", None), "schedule_deviations", None) or []
                for d in devs:
                    writer.writerow({
                        "tram_id":      tram_id,
                        "stop_id":      self._safe_int(d.get("stop_id", 0)),
                        "planned_time": round(self._safe_float(d.get("planned_time")), 4),
                        "actual_time":  round(self._safe_float(d.get("actual_time")), 4),
                        "headway_error_min": round(self._safe_float(d.get("headway_error_min")), 4),
                    })
                    total += 1

        log.info(f"Schedule deviations: {total} записей → {filepath.name}")
        return filepath