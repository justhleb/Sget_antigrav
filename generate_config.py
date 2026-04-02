"""
Генератор JSON конфигураций для имитационной модели трамваев
"""

import json
import os
import random
from typing import List, Optional, Tuple

CONFIG_DIR = "configs"


# ─── Утилиты ──────────────────────────────────────────────────────────────────

def ensure_config_dir():
    """Создаёт папку для конфигураций, если её нет"""
    if not os.path.exists(CONFIG_DIR):
        os.makedirs(CONFIG_DIR)
        print(f"Создана папка для конфигураций: {CONFIG_DIR}/")


# ─── Генераторы параметров ────────────────────────────────────────────────────

def generate_intensity_data(
    stop_number: int,
    base_intensities: Optional[List[int]] = None,
) -> List[List[int]]:
    """Генерирует данные об интенсивности прибытия пассажиров.

    Если base_intensities не задан — рассчитывается автоматически:
    центральные остановки получают более высокую базовую нагрузку.
    Почасовые коэффициенты отражают типичный суточный профиль будних дней.
    """
    if base_intensities is None:
        base_intensities = []
        center = stop_number / 2
        for i in range(1, stop_number + 1):
            distance_from_center = abs(i - center)
            base = 20 + int((stop_number - distance_from_center) * 2)
            base_intensities.append(base)

    # Коэффициенты времени суток (0.0 = нет пассажиров, 2.8 = час пик)
    time_coefficients = {
        0: 0.00, 1: 0.00, 2: 0.00, 3: 0.00, 4: 0.00, 5: 0.00,
        6: 0.30,   # Раннее утро
        7: 2.00,   # Утренний час пик
        8: 2.50,   # Пик утренний (максимум)
        9: 1.20,   # Спад после утреннего пика
        10: 0.90, 11: 0.90, 12: 1.00,  # Дневное затишье + обед
        13: 0.90, 14: 0.80, 15: 0.90, 16: 1.20,
        17: 2.20,  # Вечерний час пик
        18: 2.80,  # Вечерний пик (максимум)
        19: 1.50,  # Спад после вечернего пика
        20: 0.80, 21: 0.50, 22: 0.20, 23: 0.05,
    }

    intensity_data = []
    for stop_id in range(1, stop_number + 1):
        base = base_intensities[stop_id - 1]
        for hour in range(24):
            coef = time_coefficients.get(hour, 0.5)
            intensity_data.append([stop_id, hour, int(base * coef)])

    return intensity_data


def generate_bus_intervals(
    schedule: Optional[List[Tuple[int, int]]] = None,
) -> List[List[int]]:
    """Генерирует расписание интервалов выхода трамваев.

    Формат: [[час_начала, интервал_в_минутах], ...]
    Если schedule не задан — используется стандартный суточный профиль.
    """
    if schedule is not None:
        return schedule

    return [
        [6,  18],   # Раннее утро — редко
        [8,   8],   # Утренний час пик — часто
        [10, 15],   # День — умеренно
        [17,  8],   # Вечерний час пик — часто
        [20, 20],   # Поздний вечер — редко
        [23, 30],   # Ночь — очень редко
    ]


def generate_road_loads(
    peak_hours: Optional[List[int]] = None,
) -> List[List[float]]:
    """Генерирует данные о загруженности дорог по часам.

    Формат: [[час, коэффициент_0_до_1], ...]
    Коэффициент 0.9 = сильные пробки, 0.1 = свободно.
    """
    if peak_hours is None:
        peak_hours = [8, 9, 17, 18]

    road_loads = []
    for hour in range(24):
        if hour in peak_hours:
            load = 0.90   # Пиковые пробки
        elif hour in range(7, 10) or hour in range(16, 20):
            load = 0.70   # Полупиковая загруженность
        elif hour in range(10, 16):
            load = 0.60   # Дневная загруженность
        elif hour in range(20, 23) or hour == 6:
            load = 0.40   # Вечерний спад
        else:
            load = 0.10   # Ночь — свободно
        road_loads.append([hour, load])

    return road_loads


def generate_distances(
    stop_number: int,
    min_distance: int = 400,
    max_distance: int = 800,
    seed: Optional[int] = None,
) -> List[List[int]]:
    """Генерирует расстояния между остановками (в метрах).

    Первая остановка всегда имеет расстояние 0 (начальная точка).
    Остальные расстояния округляются до 50 метров.
    """
    rng = random.Random(seed)
    distances = [[1, 0]]
    for i in range(2, stop_number + 1):
        distance = rng.randint(min_distance, max_distance)
        distance = (distance // 50) * 50   # округление до 50 м
        distances.append([i, distance])
    return distances


# ─── Сборка и сохранение конфига ─────────────────────────────────────────────

def save_config_compact(config: dict, output_file: str):
    """Сохраняет конфигурацию в читаемом компактном JSON-формате.

    Поля intensity записываются по одной остановке на строку,
    остальные массивы — в одну строку.
    """
    ensure_config_dir()

    if not output_file.startswith(CONFIG_DIR):
        output_file = os.path.join(CONFIG_DIR, os.path.basename(output_file))

    stop_number = config["stop_number"]
    lines = ["{"]

    # ── Скалярные поля ────────────────────────────────────────────────────────
    lines.append(f'  "stop_number": {config["stop_number"]},')

    # ── Массивы одной строкой ─────────────────────────────────────────────────
    lines.append(f'  "distance": {json.dumps(config["distance"])},')

    # ── Intensity: по одной остановке на строку ───────────────────────────────
    lines.append('  "intensity": [')
    for stop_id in range(1, stop_number + 1):
        stop_data = [item for item in config["intensity"] if item[0] == stop_id]
        items_str = ", ".join(json.dumps(item) for item in stop_data)
        comma = "," if stop_id < stop_number else ""
        lines.append(f"    {items_str}{comma}")
    lines.append("  ],")

    lines.append(f'  "bus_interval": {json.dumps(config["bus_interval"])},')
    lines.append(f'  "road_loads": {json.dumps(config["road_loads"])},')

    # ── Параметры симуляции ───────────────────────────────────────────────────
    lines.append(f'  "flow_speed": {config["flow_speed"]},')
    lines.append(f'  "peak_stop": {config["peak_stop"]},')
    lines.append(f'  "tram_capacity": {config["tram_capacity"]},')
    lines.append(f'  "tram_count": {config["tram_count"]},')
    lines.append(f'  "operation_start_hour": {config["operation_start_hour"]},')
    lines.append(f'  "operation_end_hour": {config["operation_end_hour"]},')
    lines.append(f'  "simulation_hours": {config["simulation_hours"]},')
    lines.append(f'  "acceleration_time": {config["acceleration_time"]},')
    lines.append(f'  "stop_time": {config["stop_time"]},')
    lines.append(f'  "turnaround_time": {config["turnaround_time"]},')
    lines.append(f'  "target_utilization": {config["target_utilization"]},')
    lines.append(f'  "random_seed": {json.dumps(config["random_seed"])}')  # может быть null

    lines.append("}")

    with open(output_file, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"Конфигурация сохранена: {output_file}")


def create_config(
    stop_number: int = 10,
    tram_capacity: int = 90,
    tram_count: int = 8,
    simulation_hours: int = 24,
    flow_speed: int = 40,
    peak_stop: Optional[int] = None,
    operation_start_hour: int = 6,
    operation_end_hour: int = 24,
    acceleration_time: float = 0.5,
    stop_time: float = 1.0,
    turnaround_time: float = 2.0,
    target_utilization: float = 0.75,
    random_seed: Optional[int] = None,
    output_file: str = "tram_config.json",
) -> dict:
    """Создаёт полную конфигурацию и сохраняет её в файл.

    Returns:
        Словарь с конфигурацией (для использования в коде без чтения файла).
    """
    if peak_stop is None:
        peak_stop = (stop_number + 1) // 2

    config = {
        "stop_number":          stop_number,
        "distance":             generate_distances(stop_number, seed=random_seed),
        "intensity":            generate_intensity_data(stop_number),
        "bus_interval":         generate_bus_intervals(),
        "road_loads":           generate_road_loads(),
        "flow_speed":           flow_speed,
        "peak_stop":            peak_stop,
        "tram_capacity":        tram_capacity,
        "tram_count":           tram_count,
        "operation_start_hour": operation_start_hour,
        "operation_end_hour":   operation_end_hour,
        "simulation_hours":     simulation_hours,
        "acceleration_time":    acceleration_time,
        "stop_time":            stop_time,
        "turnaround_time":      turnaround_time,
        "target_utilization":   target_utilization,
        "random_seed":          random_seed,
    }

    save_config_compact(config, output_file)
    return config


# ─── Точка входа ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("Генератор конфигураций для модели трамваев")
    print("=" * 60)
    print()

    # Стандартная конфигурация — базовый сценарий для тестирования
    print("Создаём стандартную конфигурацию (10 остановок, 8 трамваев)...")
    create_config(
        stop_number=10,
        tram_capacity=90,
        tram_count=8,
        simulation_hours=24,
        random_seed=42,
        output_file="tram_config.json",
    )
    print()

    # Нагруженная конфигурация — для стресс-теста и подбора парка
    print("Создаём нагруженную конфигурацию (15 остановок, 12 трамваев)...")
    create_config(
        stop_number=15,
        tram_capacity=120,
        tram_count=12,
        simulation_hours=24,
        flow_speed=35,          # чуть медленнее — длиннее маршрут
        random_seed=42,
        output_file="tram_config_large.json",
    )
    print()

    # Минималистичная конфигурация — быстрые прогоны при разработке
    print("Создаём минималистичную конфигурацию (6 остановок, 4 трамвая)...")
    create_config(
        stop_number=6,
        tram_capacity=60,
        tram_count=4,
        simulation_hours=12,    # полдня — достаточно для отладки
        random_seed=42,
        output_file="tram_config_small.json",
    )
    print()

    print("=" * 60)
    print(f"✓ Все конфигурации сохранены в папку: {CONFIG_DIR}/")
    print("=" * 60)
