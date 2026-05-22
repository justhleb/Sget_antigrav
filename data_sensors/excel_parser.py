"""
Парсер агрегированной финансовой статистики из Excel файлов (директория summary_reports).

Этот модуль считывает экономические показатели по каждому трамвайному маршруту из отчетов Excel.
Каждый файл содержит статистику по конкретным транспортным средствам (ТС) на маршруте:
- Количество перевезенных пассажиров за час и на 1 км пробега.
- Полученные доходы за час и на 1 км пробега.

Результатом работы является словарь, сопоставляющий ID маршрута с агрегированными 
экономическими показателями (средними по всем ТС).
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

# Дефолтный путь к папке с отчетами относительно текущего файла
SUMMARY_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "summary_reports")


@dataclass
class VehicleEconomics:
    """
    Класс для хранения экономической статистики одного транспортного средства (ТС) на маршруте.
    """
    vehicle_id: str             # Уникальный идентификатор/номер ТС
    passengers_per_hour: float  # Среднее количество перевезенных пассажиров в час
    passengers_per_km: float    # Среднее количество перевезенных пассажиров на 1 км пробега
    revenue_per_hour: float     # Средний доход в час (руб.)
    revenue_per_km: float       # Средний доход на 1 км пробега (руб.)


@dataclass
class RouteEconomics:
    """
    Класс для хранения агрегированной экономической статистики по всему маршруту.
    Вычисляет средние удельные показатели на основе данных от всех ТС, работающих на маршруте.
    """
    route_id: str                                    # Идентификатор маршрута (например, "20", "48", "55")
    vehicles: List[VehicleEconomics] = field(default_factory=list) # Список статистики по всем ТС

    # ── Средние удельные показатели по всем ТС маршрута ───────────────────────
    
    @property
    def mean_passengers_per_km(self) -> float:
        """Среднее число пассажиров на 1 км пробега по всем ТС."""
        if not self.vehicles:
            return 0.0
        return sum(v.passengers_per_km for v in self.vehicles) / len(self.vehicles)

    @property
    def mean_revenue_per_km(self) -> float:
        """Средний доход на 1 км пробега (в рублях) по всем ТС."""
        if not self.vehicles:
            return 0.0
        return sum(v.revenue_per_km for v in self.vehicles) / len(self.vehicles)

    @property
    def mean_passengers_per_hour(self) -> float:
        """Среднее число пассажиров в час по всем ТС."""
        if not self.vehicles:
            return 0.0
        return sum(v.passengers_per_hour for v in self.vehicles) / len(self.vehicles)

    @property
    def mean_revenue_per_hour(self) -> float:
        """Средний доход в час (в рублях) по всем ТС."""
        if not self.vehicles:
            return 0.0
        return sum(v.revenue_per_hour for v in self.vehicles) / len(self.vehicles)


def _extract_route_id(filename: str) -> str:
    """
    Извлекает идентификатор маршрута из имени файла.
    Пример: "summary_report_20.xlsx" -> "20"
    
    :param filename: Имя или путь к файлу.
    :return: Строковый идентификатор маршрута.
    """
    m = re.search(r"summary_report_(\d+)", filename)
    return m.group(1) if m else os.path.splitext(os.path.basename(filename))[0]


def load_route_economics(
    summary_dir: str = SUMMARY_DIR,
) -> Dict[str, RouteEconomics]:
    """
    Загружает экономические данные из всех Excel-файлов в указанной директории summary_dir.
    Ожидаются файлы с маской summary_report_*.xlsx.
    
    Каждый файл парсится с помощью pandas. Файлы имеют двухуровневую шапку:
      (№ п/п, ТС) | (Пассажиры, чел. -> На 1 час/На 1 км) | (Доходы, руб. -> На 1 час/На 1 км)

    :param summary_dir: Путь к директории с отчетами Excel.
    :return: Словарь, где ключ — строковый ID маршрута, а значение — объект RouteEconomics.
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

        # Считываем Excel-файл с мультииндексом в качестве заголовка (двухуровневая шапка)
        df = pd.read_excel(filepath, header=[0, 1])

        # Нормализуем имена столбцов и проверяем структуру
        # Ожидаемый порядок столбцов: ТС, Пассажиры/час, Пассажиры/км, Доходы/час, Доходы/км
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

            # Фильтруем пустые или аномальные строки, где нет ни пробега, ни доходов
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
