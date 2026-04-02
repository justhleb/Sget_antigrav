"""
Парсер агрегированной финансовой статистики из Excel файлов (summary_reports).

Каждый файл — один маршрут. Двухуровневые заголовки:
  (№ п/п, ТС) | (Пассажиры, чел. → На 1 час/На 1 км) | (Доходы, руб. → На 1 час/На 1 км)

Результат: словарь {route_id: RouteEconomics} со средними удельными показателями.
"""
from __future__ import annotations

import glob
import logging
import os
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import pandas as pd

log = logging.getLogger(__name__)

SUMMARY_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "summary_reports")


@dataclass
class VehicleEconomics:
    """Статистика одного ТС на конкретном маршруте."""
    vehicle_id: str
    passengers_per_hour: float
    passengers_per_km: float
    revenue_per_hour: float   # руб.
    revenue_per_km: float      # руб.


@dataclass
class RouteEconomics:
    """Агрегированная экономическая статистика маршрута."""
    route_id: str
    vehicles: List[VehicleEconomics] = field(default_factory=list)

    # ── средние по всем ТС ────────────────────────────────────────────────────
    @property
    def mean_passengers_per_km(self) -> float:
        if not self.vehicles:
            return 0.0
        return sum(v.passengers_per_km for v in self.vehicles) / len(self.vehicles)

    @property
    def mean_revenue_per_km(self) -> float:
        if not self.vehicles:
            return 0.0
        return sum(v.revenue_per_km for v in self.vehicles) / len(self.vehicles)

    @property
    def mean_passengers_per_hour(self) -> float:
        if not self.vehicles:
            return 0.0
        return sum(v.passengers_per_hour for v in self.vehicles) / len(self.vehicles)

    @property
    def mean_revenue_per_hour(self) -> float:
        if not self.vehicles:
            return 0.0
        return sum(v.revenue_per_hour for v in self.vehicles) / len(self.vehicles)


def _extract_route_id(filename: str) -> str:
    """summary_report_20.xlsx → '20'"""
    m = re.search(r"summary_report_(\d+)", filename)
    return m.group(1) if m else os.path.splitext(os.path.basename(filename))[0]


def load_route_economics(
    summary_dir: str = SUMMARY_DIR,
) -> Dict[str, RouteEconomics]:
    """
    Загружает все xlsx файлы из summary_dir.

    Returns
    -------
    Dict[str, RouteEconomics]
        Ключ — route_id (строка), значение — агрегированная статистика.
    """
    result: Dict[str, RouteEconomics] = {}
    pattern = os.path.join(summary_dir, "summary_report_*.xlsx")
    files = sorted(glob.glob(pattern))

    if not files:
        log.warning(f"Не найдено файлов summary_report_*.xlsx в {summary_dir}")
        return result

    for filepath in files:
        route_id = _extract_route_id(filepath)
        log.info(f"Загрузка экономических данных маршрута {route_id}: {filepath}")

        df = pd.read_excel(filepath, header=[0, 1])

        # Нормализуем имена столбцов
        # Ожидаемый порядок: ТС, Пасс/час, Пасс/км, Доход/час, Доход/км
        cols = df.columns.tolist()
        if len(cols) < 5:
            log.warning(f"Файл {filepath}: ожидалось ≥5 столбцов, получено {len(cols)} — пропуск")
            continue

        vehicles: List[VehicleEconomics] = []
        for _, row in df.iterrows():
            vehicle_id = str(row.iloc[0]).strip()
            try:
                pax_hour = float(row.iloc[1])
                pax_km   = float(row.iloc[2])
                rev_hour = float(row.iloc[3])
                rev_km   = float(row.iloc[4])
            except (ValueError, TypeError):
                log.warning(f"Маршрут {route_id}: пропуск строки с ТС '{vehicle_id}' (нечисловые данные)")
                continue

            # Фильтруем аномальные/пустые строки
            if pax_km <= 0 and rev_km <= 0:
                continue

            vehicles.append(VehicleEconomics(
                vehicle_id=vehicle_id,
                passengers_per_hour=pax_hour,
                passengers_per_km=pax_km,
                revenue_per_hour=rev_hour,
                revenue_per_km=rev_km,
            ))

        route_econ = RouteEconomics(route_id=route_id, vehicles=vehicles)
        result[route_id] = route_econ
        log.info(
            f"  Маршрут {route_id}: {len(vehicles)} ТС, "
            f"ср. доход/км = {route_econ.mean_revenue_per_km:.2f} руб., "
            f"ср. пасс/км = {route_econ.mean_passengers_per_km:.2f}"
        )

    return result
