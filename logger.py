"""
Модуль логирования детальных событий трамваев.
"""
from __future__ import annotations

import csv
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any

log = logging.getLogger(__name__)


class TramLogger:

    TRAM_LOG_COLUMNS = [
        "tram_id",
        "route_id",
        "trip_id",           # ✅ новое
        "time_min",
        "planned_time",      # ✅ новое
        "headway_error_min", # ✅ замена delay_min
        "hour",
        "stop_id",
        "direction",
        "waiting_before",
        "alighted",
        "boarded",
        "passengers_in_tram",
        "utilization_percent",
    ]

    SUMMARY_COLUMNS = [
        "tram_id",
        "route_id",
        "total_trips",
        "passengers_served",
        "avg_utilization_percent",
        "max_utilization_percent",
        "avg_headway_error_min", # ✅ замена delay_min
        "max_headway_error_min", # ✅ замена delay_min
        "stop_events_count",
    ]

    DEVIATION_COLUMNS = [   # ✅ новый CSV
        "tram_id",
        "route_id",
        "stop_id",
        "planned_time",
        "actual_time",
        "headway_error_min",
    ]

    def __init__(
        self,
        output_dir: str = "tram_logs",
        file_prefix: str = "tram",
        write_header: bool = True,
    ):
        self.output_dir   = Path(output_dir)
        self.file_prefix  = file_prefix
        self.write_header = write_header
        self._ensure_output_dir()

    def _ensure_output_dir(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _direction_label(direction: str) -> str:
        return direction if direction in ("forward", "backward") else str(direction)

    @staticmethod
    def _safe_int(x: Any, default: int = 0) -> int:
        try:    return int(x)
        except: return default

    @staticmethod
    def _safe_float(x: Any, default: float = 0.0) -> float:
        try:    return float(x)
        except: return default

    # ── Лог остановок трамвая ─────────────────────────────────────────────────

    def save_tram_log(
        self,
        tram_id: int,
        stop_log: List[dict],
        route_id: Optional[str] = None,
    ) -> Optional[Path]:
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
                planned  = event.get("planned_time")
                headway_error = event.get("headway_error_min")
                writer.writerow({
                    "tram_id":            tram_id,
                    "route_id":           event.get("route_id", route_id),
                    "trip_id":            self._safe_int(event.get("trip_id", 0)),
                    "time_min":           round(t, 4),
                    "planned_time":       round(float(planned), 4) if planned is not None else "",
                    "headway_error_min":  round(float(headway_error), 4) if headway_error is not None else "",
                    "hour":               int(t // 60) % 24,
                    "stop_id":            self._safe_int(event.get("stop_id", 0)),
                    "direction":          self._direction_label(event.get("direction", "")),
                    "waiting_before":     self._safe_int(event.get("waiting_before", 0)),
                    "alighted":           self._safe_int(event.get("alighted", 0)),
                    "boarded":            self._safe_int(event.get("boarded", 0)),
                    "passengers_in_tram": self._safe_int(event.get("passengers_in_tram", 0)),
                    "utilization_percent": round(self._safe_float(event.get("utilization_after", 0.0)), 4),
                })

        log.info(f"Tram #{tram_id}: {len(stop_log)} stop events → {filepath.name}")
        return filepath

    def save_all_trams(
        self,
        trams: Dict[int, Any],
        route_id: Optional[str] = None,
        include_empty: bool = False,
    ) -> List[Path]:
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

    # ── Сводная таблица ───────────────────────────────────────────────────────

    def create_summary(
        self,
        trams: Dict[int, Any],
        output_file: str = "trams_summary.csv",
        route_id: Optional[str] = None,
    ) -> Path:
        filepath = self.output_dir / output_file

        with filepath.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=self.SUMMARY_COLUMNS)
            writer.writeheader()

            for tram_id, tram in sorted(trams.items(), key=lambda x: x[0]):
                stats = getattr(tram, "stats", None)
                if stats is None:
                    continue

                util_hist = list(getattr(stats, "utilization_history", []) or [])
                stop_log  = list(getattr(stats, "stop_log",  []) or [])

                if not util_hist and not stop_log:
                    continue

                avg_util = (sum(util_hist) / len(util_hist)) if util_hist else 0.0
                max_util = max(util_hist) if util_hist else 0.0

                # Считаем ошибки интервалов из stop_log
                errors = [
                    float(e["headway_error_min"]) for e in stop_log
                    if e.get("headway_error_min") not in (None, "")
                ]
                avg_error = round(sum(errors) / len(errors), 4) if errors else 0.0
                max_error = round(max(errors), 4) if errors else 0.0

                detected_route_id = stop_log[0].get("route_id") if stop_log else None

                writer.writerow({
                    "tram_id":                 tram_id,
                    "route_id":                detected_route_id or route_id,
                    "total_trips":             int(getattr(stats, "total_trips", 0) or 0),
                    "passengers_served":       int(getattr(stats, "passengers_served", 0) or 0),
                    "avg_utilization_percent": round(avg_util * 100.0, 4),
                    "max_utilization_percent": round(max_util * 100.0, 4),
                    "avg_headway_error_min":   avg_error,
                    "max_headway_error_min":   max_error,
                    "stop_events_count":       len(stop_log),
                })

        log.info(f"Summary saved → {filepath.name}")
        return filepath

    # ── Отклонения от расписания (отдельный CSV) ──────────────────────────────

    def save_schedule_deviations(
        self,
        trams: Dict[int, Any],
        output_file: str = "schedule_deviations.csv",
        route_id: Optional[str] = None,
    ) -> Path:
        """
        Сохраняет все отклонения от расписания по всем трамваям в один CSV.
        Удобно для анализа — можно загрузить в pandas и сразу считать метрики.
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
                        "route_id":     d.get("route_id", route_id),
                        "stop_id":      self._safe_int(d.get("stop_id", 0)),
                        "planned_time": round(self._safe_float(d.get("planned_time")), 4),
                        "actual_time":  round(self._safe_float(d.get("actual_time")), 4),
                        "headway_error_min": round(self._safe_float(d.get("headway_error_min")), 4),
                    })
                    total += 1

        log.info(f"Schedule deviations: {total} записей → {filepath.name}")
        return filepath