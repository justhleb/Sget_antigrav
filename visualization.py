"""
Модуль визуализации результатов симуляции трамваев.

Этот модуль отвечает за построение наглядных графиков по итогам работы симуляции:
1. Графиков качества движения (ошибки интервалов по часам, по остановкам, тепловая карта).
2. Сводного финансового дашборда для оценки экономической эффективности маршрутов (выручка, расходы, маржинальность, CMR).

Для построения используется библиотека matplotlib.
"""

from __future__ import annotations

import logging
import traceback
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

# Используем бэкенд Agg для генерации графиков без вывода на экран (удобно для серверов и фонового запуска)
matplotlib.use("Agg")
plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans", "sans-serif"]
plt.rcParams["axes.unicode_minus"] = False

log = logging.getLogger(__name__)

# ─── Дефолтные константы оформления ──────────────────────────────────────────
_DEFAULT_TARGET_UTIL   = 0.75                  # Целевой коэффициент загрузки трамвая (не используется напрямую в расчетах)
_DEFAULT_COMFORT_LOW   = 0.60                  # Нижний порог комфортной загрузки пассажирами
_DEFAULT_COMFORT_HIGH  = 0.80                  # Верхний порог комфортной загрузки пассажирами
_DEFAULT_OVERLOAD      = 0.90                  # Порог перегрузки трамвая
_DEFAULT_PEAK_RANGES   = [(7, 9), (17, 19)]    # Временные интервалы часов пик (утро и вечер)
_DPI                   = 150                   # Разрешение сохраняемых картинок графиков
HOUR_START  = 5                                # Начало операционного дня (5:00 утра)
HOUR_END    = 24                               # Конец операционного дня (24:00)
_OP_HOURS   = list(range(HOUR_START, HOUR_END))# Список рабочих часов симуляции


class TramVisualization:
    """
    Класс для визуализации пространственно-временных результатов симуляции.
    Строит графики ошибок интервалов по остановкам, по часам и в виде тепловой карты.
    """

    def __init__(
        self,
        stops: Dict,
        simulation_hours: int = 24,
        target_utilization: float = _DEFAULT_TARGET_UTIL,
        comfort_low: float = _DEFAULT_COMFORT_LOW,
        comfort_high: float = _DEFAULT_COMFORT_HIGH,
        overload_threshold: float = _DEFAULT_OVERLOAD,
        peak_hour_ranges: Optional[List[Tuple[int, int]]] = None,
        route_id: Optional[str] = None,
    ):
        """
        Инициализирует визуализатор для конкретного маршрута.

        :param stops: Словарь со всеми остановками маршрута.
        :param simulation_hours: Продолжительность симуляции в часах.
        :param target_utilization: Целевая загрузка.
        :param comfort_low: Нижняя граница комфортной загрузки.
        :param comfort_high: Верхняя граница комфортной загрузки.
        :param overload_threshold: Порог перегрузки.
        :param peak_hour_ranges: Границы пиковых часов.
        :param route_id: Уникальный идентификатор маршрута (для заголовков графиков).
        """
        self.stops = stops
        self.simulation_hours = simulation_hours

        # Сортируем ID остановок для правильного порядка на осях графиков
        self.stop_ids: List[int] = sorted(stops.keys())
        self.stop_number: int = len(self.stop_ids)

        # Сопоставляем глобальные ID остановок с их локальными порядковыми номерами (от 1)
        self.stop_labels: Dict[int, int] = {
            gid: local
            for local, gid in enumerate(self.stop_ids, start=1)
        }

        self.target_util   = target_utilization
        self.comfort_low   = comfort_low
        self.comfort_high  = comfort_high
        self.overload      = overload_threshold
        self.peak_ranges   = peak_hour_ranges or _DEFAULT_PEAK_RANGES
        self.route_id      = route_id

    # ── Вспомогательные методы оформления ─────────────────────────────────────

    def _title(self, base: str) -> str:
        """Формирует заголовок графика с указанием номера маршрута, если он задан."""
        return f"{base} (маршрут {self.route_id})" if self.route_id else base

    def _add_peak_spans(self, ax: plt.Axes) -> None:
        """Подсвечивает зоны пиковых часов на графике фоновым цветом (красный/оранжевый)."""
        colors = ["red", "orange", "red", "orange"]
        for i, (h_start, h_end) in enumerate(self.peak_ranges):
            ax.axvspan(h_start, h_end, alpha=0.10, color=colors[i % len(colors)])

    def _add_peak_labels(self, ax: plt.Axes, y: float) -> None:
        """Добавляет текстовые метки ('Утренний час пик' и т.д.) над пиковыми зонами."""
        default_labels = ["Утренний\nчас пик", "Вечерний\nчас пик"]
        for i, (h_start, h_end) in enumerate(self.peak_ranges):
            label = default_labels[i] if i < len(default_labels) else f"Пик {i+1}"
            ax.text(
                (h_start + h_end) / 2, y, label,
                ha="center", fontsize=9, alpha=0.7
            )

    @staticmethod
    def _collect_deviations(trams: Dict) -> List[dict]:
        """
        Собирает все записи об отклонениях (schedule_deviations) 
        из всех переданных трамваев в один плоский список.
        """
        result = []
        for tram in trams.values():
            devs = getattr(getattr(tram, "stats", None), "schedule_deviations", None) or []
            result.extend(devs)
        return result

    # ── Графики отклонений от расписания ─────────────────────────────────────

    def plot_delay_by_stop(
        self,
        trams: Dict,
        output_file: str | Path = "delay_by_stop.png",
    ) -> Path:
        """
        Строит столбчатую диаграмму (bar chart) со средней ошибкой интервалов по каждой остановке.
        Позволяет визуально определить, на каких остановках/участках трамваи систематически сбиваются с темпа.
        
        :param trams: Словарь с объектами трамваев симуляции.
        :param output_file: Путь к файлу для сохранения графика.
        """
        output_file = Path(output_file)
        log.info("Создание графика ошибок интервалов по остановкам...")

        deviations = self._collect_deviations(trams)
        if not deviations:
            log.warning("Нет данных об ошибках интервалов — пропускаем")
            return output_file

        # Группируем значения ошибок интервалов headway_error_min по stop_id
        stop_delays: Dict[int, List[float]] = {}
        for d in deviations:
            sid = d["stop_id"]
            if d.get("headway_error_min") is not None:
                stop_delays.setdefault(sid, []).append(d["headway_error_min"])

        # Фильтруем и упорядочиваем по списку остановок маршрута
        ordered_ids = [sid for sid in self.stop_ids if sid in stop_delays]
        labels      = [str(self.stop_labels[sid]) for sid in ordered_ids]
        means       = [float(np.mean(stop_delays[sid])) for sid in ordered_ids]
        stds        = [float(np.std(stop_delays[sid]))  for sid in ordered_ids]

        # Подсвечиваем красным (tomato), если средняя ошибка > 5 минут, иначе синим (steelblue)
        colors = ["tomato" if m > 5 else "steelblue" for m in means]

        fig, ax = plt.subplots(figsize=(max(12, len(labels) * 0.6), 6))
        bars = ax.bar(labels, means, color=colors, alpha=0.8,
                      edgecolor="black", linewidth=0.5)
        
        # Рисуем стандартное отклонение в виде «усов»
        ax.errorbar(labels, means, yerr=stds, fmt="none",
                    color="black", capsize=4, linewidth=1.2, alpha=0.6)

        # Добавляем горизонтальные линии для индикации порогов критичности
        ax.axhline(0, color="black", linewidth=1.5, linestyle="-")
        ax.axhline(2,  color="orange", linewidth=1, linestyle="--",
                   alpha=0.7, label="2 мин (нормально)")
        ax.axhline(5,  color="red", linewidth=1, linestyle="--",
                   alpha=0.7, label="5 мин (критично)")

        ax.set_title(self._title("Средняя ошибка интервалов по остановкам"),
                     fontsize=14, fontweight="bold")
        ax.set_xlabel("Номер остановки", fontsize=12)
        ax.set_ylabel("Ошибка интервала (мин)", fontsize=11)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3, axis="y", linestyle="--")

        # Подписываем конкретные значения над каждым столбцом диаграммы
        for bar, mean in zip(bars, means):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                mean + 0.15,
                f"{mean:.1f}",
                ha="center", va="bottom", fontsize=8,
            )

        plt.tight_layout()
        plt.savefig(output_file, dpi=_DPI, bbox_inches="tight")
        plt.close()
        log.info(f"График отклонений по остановкам сохранён: {output_file.name}")
        return output_file

    def plot_delay_by_hour(
        self,
        trams: Dict,
        output_file: str | Path = "delay_by_hour.png",
    ) -> Path:
        """
        Строит линейный график средней ошибки интервалов по часам суток.
        Показывает динамику сбивания расписания, отражающую влияние дорожной загрузки в течение дня.

        :param trams: Словарь с объектами трамваев симуляции.
        :param output_file: Путь к файлу для сохранения графика.
        """
        output_file = Path(output_file)
        log.info("Создание графика ошибок интервалов по часам...")

        deviations = self._collect_deviations(trams)
        if not deviations:
            log.warning("Нет данных об ошибках интервалов — пропускаем")
            return output_file

        # Группируем ошибки по часам суток (на основе фактического времени прибытия actual_time)
        hourly_delays: Dict[int, List[float]] = {h: [] for h in range(24)}
        for d in deviations:
            if d.get("headway_error_min") is not None:
                hour = int(d.get("actual_time", 0) // 60) % 24
                hourly_delays[hour].append(d["headway_error_min"])

        hours     = _OP_HOURS
        means     = [float(np.mean(hourly_delays[h])) if hourly_delays[h] else 0.0
                     for h in hours]
        stds      = [float(np.std(hourly_delays[h]))  if hourly_delays[h] else 0.0
                     for h in hours]
        means_arr = np.array(means)
        stds_arr  = np.array(stds)

        fig, ax = plt.subplots(figsize=(14, 6))

        # Отрисовываем основную линию средних значений
        ax.plot(hours, means, linewidth=2.5, color="steelblue",
                marker="o", markersize=6, label="Средняя ошибка интервала")
        
        # Подсвечиваем полупрозрачной областью диапазон ±1 стандартное отклонение (σ)
        ax.fill_between(hours,
                        np.clip(means_arr - stds_arr, 0, None),
                        means_arr + stds_arr,
                        alpha=0.2, color="steelblue", label="±σ (разброс)")

        ax.axhline(0, color="black", linewidth=1.5)
        ax.axhline(2,  color="orange", linewidth=1, linestyle="--",
                   alpha=0.8, label="2 мин (допустимо)")
        ax.axhline(5,  color="red", linewidth=1, linestyle="--",
                   alpha=0.8, label="5 мин (критично)")

        # Подсвечиваем утренний и вечерний пиковые интервалы
        self._add_peak_spans(ax)
        y_label = max(means) * 0.85 if max(means) > 0 else 1.0
        self._add_peak_labels(ax, y_label)

        ax.set_title(self._title("Средняя ошибка интервалов по часам суток"),
                     fontsize=14, fontweight="bold")
        ax.set_xlabel("Час дня", fontsize=12)
        ax.set_ylabel("Ошибка интервала (мин)", fontsize=11)
        ax.set_xticks(hours)
        ax.grid(True, alpha=0.3, linestyle="--")
        ax.legend(fontsize=10)

        plt.tight_layout()
        plt.savefig(output_file, dpi=_DPI, bbox_inches="tight")
        plt.close()
        log.info(f"График отклонений по часам сохранён: {output_file.name}")
        return output_file

    def plot_delay_heatmap(
        self,
        trams: Dict,
        output_file: str | Path = "delay_heatmap.png",
    ) -> Path:
        """
        Строит двумерную тепловую карту (heat map):
        - Ось X: часы суток (операционное время).
        - Ось Y: последовательность остановок маршрута.
        - Цвет: средняя ошибка интервалов (интенсивность красного цвета указывает на размер сбоя).
        
        Это наиболее детальный график, позволяющий мгновенно локализовать «бутылочные горлышки» — 
        какие остановки в какие часы вызывают наибольшие задержки движения.

        :param trams: Словарь с объектами трамваев симуляции.
        :param output_file: Путь к файлу для сохранения графика.
        """
        output_file = Path(output_file)
        log.info("Создание тепловой карты ошибок интервалов...")

        deviations = self._collect_deviations(trams)
        if not deviations:
            log.warning("Нет данных об ошибках интервалов — пропускаем")
            return output_file

        # Матрица: строки — остановки, столбцы — часы дня
        delay_matrix: Dict[int, Dict[int, List[float]]] = {
            sid: {h: [] for h in range(24)} for sid in self.stop_ids
        }
        for d in deviations:
            if d.get("headway_error_min") is not None:
                sid  = d["stop_id"]
                hour = int(d.get("actual_time", 0) // 60) % 24
                if sid in delay_matrix:
                    delay_matrix[sid][hour].append(d["headway_error_min"])

        # Заполняем массив numpy средними значениями задержек
        data = np.zeros((self.stop_number, len(_OP_HOURS)))
        for idx, sid in enumerate(self.stop_ids):
            for col, h in enumerate(_OP_HOURS):
                vals = delay_matrix[sid][h]
                data[idx, col] = float(np.mean(vals)) if vals else 0.0

        vmax = max(data.max(), 1.0)

        fig, ax = plt.subplots(figsize=(14, max(8, self.stop_number * 0.4)))
        # Рисуем карту с палитрой "OrRd" (Orange-Red: от белого через оранжевый к темно-красному)
        im = ax.imshow(data, cmap="OrRd", aspect="auto",
                       interpolation="nearest", vmin=0, vmax=vmax)

        ax.set_xticks(range(len(_OP_HOURS)))
        ax.set_xticklabels(_OP_HOURS)
        ax.set_yticks(range(self.stop_number))
        ax.set_yticklabels([f"Ост. {self.stop_labels[sid]}" for sid in self.stop_ids])
        
        ax.set_title(
            self._title("Тепловая карта ошибок интервалов"),
            fontsize=14, fontweight="bold",
        )
        ax.set_xlabel("Час дня", fontsize=12)
        ax.set_ylabel("Остановка", fontsize=12)

        # Выводим цветовую шкалу сбоку
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label("Ошибка интервала (мин)", fontsize=10)

        # Настраиваем сетку для разделения ячеек тепловой карты
        ax.set_xticks(np.arange(24) - 0.5, minor=True)
        ax.set_yticks(np.arange(self.stop_number) - 0.5, minor=True)
        ax.grid(which="minor", color="white", linestyle="-", linewidth=0.5)

        plt.tight_layout()
        plt.savefig(output_file, dpi=_DPI, bbox_inches="tight")
        plt.close()
        log.info(f"Тепловая карта отклонений сохранена: {output_file.name}")
        return output_file

    # ── Главный метод визуализации ────────────────────────────────────────────

    def create_all_plots(
        self,
        trams: Optional[Dict] = None,
        output_dir: str | Path = ".",
    ) -> List[Path]:
        """
        Запускает построение и сохранение всех основных графиков качества движения.

        :param trams: Словарь с объектами трамваев.
        :param output_dir: Папка для сохранения PNG-изображений.
        :return: Список путей к успешно созданным графическим файлам.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        log.info("\n" + "=" * 60)
        log.info("СОЗДАНИЕ ГРАФИКОВ")
        log.info(f"Папка: {output_dir.resolve()}")
        log.info("=" * 60)

        tasks = []
        if trams:
            tasks += [
                ("delay_by_stop.png",            lambda p: self.plot_delay_by_stop(trams, p)),
                ("delay_by_hour.png",            lambda p: self.plot_delay_by_hour(trams, p)),
                ("delay_heatmap.png",            lambda p: self.plot_delay_heatmap(trams, p)),
            ]

        created: List[Path] = []
        for filename, plot_fn in tasks:
            filepath = output_dir / filename
            try:
                result = plot_fn(filepath)
                if result is not None:
                    created.append(result)
            except Exception:
                log.error(f"Ошибка при построении {filename}:\n{traceback.format_exc()}")

        log.info("=" * 60)
        log.info(f"Создано графиков: {len(created)} / {len(tasks)}")
        log.info("=" * 60)
        return created


def plot_global_financial_summary(stats: dict, output_file: str | Path) -> Path:
    """
    Отрисовывает красивый и современный сводный финансовый дашборд (Выручка, VarEx, Опер. результат, CMR).
    Применяет стильную, гармоничную цветовую схему, закругленные углы, аккуратные сетки и Inter-подобный шрифт.
    
    Дашборд включает:
    1. Верхнюю панель с 5 основными KPI карточками (выручка, расходы, прибыль, выручка/км, CMR).
    2. График сравнения Выручки и Переменных затрат (VarEx) по маршрутам.
    3. Горизонтальную диаграмму опер. результата с подсветкой убытков красным цветом.
    4. Столбчатую диаграмму удельного опер. результата на 1 км пробега.
    5. Столбчатую диаграмму рентабельности продаж (CMR, %).

    :param stats: Словарь с глобальной и помаршрутной статистикой симуляции.
    :param output_file: Путь для сохранения итогового дашборда (PNG).
    :return: Path к созданному файлу изображения.
    """
    import matplotlib.font_manager as fm

    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    log.info("Создание финансового дашборда...")

    routes_stats = stats.get("routes", {})
    global_stats = stats.get("global", {})

    # Группируем fwd и bwd направления в единый маршрут (по числовому номеру)
    route_nums: Dict[str, dict] = {}
    for rid, rs in routes_stats.items():
        num = rid.split("_")[0]
        if num not in route_nums:
            route_nums[num] = {
                "revenue": 0, "varex": 0, "marginal_profit": 0,
                "tram_km": 0, "total_trips": 0,
            }
        route_nums[num]["revenue"] += rs.get("revenue", 0)
        route_nums[num]["varex"] += rs.get("varex", 0)
        route_nums[num]["marginal_profit"] += rs.get("marginal_profit", 0)
        route_nums[num]["tram_km"] += rs.get("tram_km", 0)
        route_nums[num]["total_trips"] += rs.get("total_trips", 0)

    for num, d in route_nums.items():
        d["profit_per_km"] = d["marginal_profit"] / d["tram_km"] if d["tram_km"] > 0 else 0
        d["cmr_pct"] = (d["marginal_profit"] / d["revenue"] * 100) if d["revenue"] > 0 else 0

    labels = sorted(route_nums.keys())
    if not labels:
        log.warning("Нет данных о маршрутах для дашборда.")
        return output_file

    # ── Подбор красивого шрифта Inter (или падение на системные шрифты) ──────────
    inter_props = {}
    for fp in fm.findSystemFonts():
        try:
            name = fm.FontProperties(fname=fp).get_name()
            if "inter" in name.lower():
                inter_props = {"fontproperties": fm.FontProperties(fname=fp)}
                break
        except Exception:
            continue

    # ── Цветовая палитра премиум-класса ───────────────────────────────────────
    C_BG       = "#FFFFFF"   # Чисто белый фон
    C_TEXT     = "#2D2D2D"   # Темно-серый текст
    C_MUTED    = "#8C8C8C"   # Приглушенный серый
    C_GRID     = "#E8E8E8"   # Ненавязчивые линии сетки
    C_REVENUE  = "#34A853"   # Изумрудный зеленый (выручка)
    C_VAREX    = "#EA4335"   # Коралловый красный (расходы/убытки)
    C_MARGIN   = "#4285F4"   # Спокойный синий (прибыль)
    C_CMR      = "#FBBC05"   # Солнечный желтый (рентабельность)
    C_PROFKM   = "#7B61FF"   # Мягкий фиолетовый (прибыль на км)
    C_BAR_REV  = "#34A853"
    C_BAR_VAREX = "#EA4335"

    # ── Layout: Сетка из 3 строк × 2 колонок ─────────────────────────────────
    fig = plt.figure(figsize=(16, 14), facecolor=C_BG)
    gs = fig.add_gridspec(
        3, 2,
        height_ratios=[0.22, 1, 1],  # KPI панель занимает 22% высоты, остальные графики поровну
        hspace=0.35, wspace=0.30,
        left=0.07, right=0.95, top=0.94, bottom=0.05,
    )

    # ── ROW 0: Панель KPI-карточек ───────────────────────────────────────────
    ax_kpi = fig.add_subplot(gs[0, :])
    ax_kpi.set_xlim(0, 1)
    ax_kpi.set_ylim(0, 1)
    ax_kpi.axis("off")

    g = global_stats
    kpis = [
        ("Выручка",            g.get("total_revenue", 0),       "₽", C_REVENUE),
        ("VarEx",              g.get("varex", 0),               "₽", C_VAREX),
        ("Опер. результат",    g.get("marginal_profit", 0),     "₽", C_MARGIN),
        ("Выручка/км",         g.get("profit_per_km", 0),       "₽/км", C_PROFKM),
        ("CMR",                g.get("cmr_pct", 0),             "%", C_CMR),
    ]

    card_w = 1.0 / len(kpis)
    for i, (title, value, unit, color) in enumerate(kpis):
        cx = card_w * i + card_w / 2
        # Рисуем заголовок карточки
        ax_kpi.text(cx, 0.82, title.upper(), fontsize=10, ha="center", va="center",
                    color=C_MUTED, fontweight="normal", **inter_props)
        # Форматируем отображение крупных чисел
        if unit == "%":
            val_str = f"{value:+.1f}{unit}"
        elif abs(value) >= 1_000_000:
            val_str = f"{value/1_000_000:,.2f} М{unit}"
        elif abs(value) >= 1_000:
            val_str = f"{value:,.0f} {unit}"
        else:
            val_str = f"{value:,.2f} {unit}"
        
        # Выводим числовое значение по центру карточки
        ax_kpi.text(cx, 0.38, val_str, fontsize=22, ha="center", va="center",
                    color=color, fontweight="bold", **inter_props)
        # Рисуем вертикальный разделитель между карточками
        if i < len(kpis) - 1:
            ax_kpi.axvline(card_w * (i + 1), color=C_GRID, linewidth=1, ymin=0.15, ymax=0.85)

    # Делаем красивую закругленную подложку для KPI-карточек
    for spine in ["top", "bottom", "left", "right"]:
        ax_kpi.spines[spine].set_visible(False)
    rect = matplotlib.patches.FancyBboxPatch(
        (0.005, 0.05), 0.99, 0.90, boxstyle="round,pad=0.02",
        facecolor="#F9F9F9", edgecolor=C_GRID, linewidth=1.5,
        transform=ax_kpi.transAxes, zorder=-1,
    )
    ax_kpi.add_patch(rect)

    # ── ROW 1: Выручка vs OpEx (сгруппированные столбцы) ────────────────────
    ax1 = fig.add_subplot(gs[1, 0], facecolor=C_BG)
    x = np.arange(len(labels))
    w = 0.35
    revs = [route_nums[l]["revenue"] for l in labels]
    varexs = [route_nums[l]["varex"] for l in labels]

    ax1.bar(x - w/2, revs,  w, color=C_BAR_REV,   alpha=0.85, label="Выручка", edgecolor="none")
    ax1.bar(x + w/2, varexs, w, color=C_BAR_VAREX, alpha=0.85, label="VarEx",   edgecolor="none")

    # Подписи значений непосредственно над каждым столбцом
    for xi, (rv, ox) in enumerate(zip(revs, varexs)):
        ax1.text(xi - w/2, rv + rv * 0.01, f"{rv:,.0f}", ha="center", va="bottom",
                 fontsize=8, color=C_TEXT, **inter_props)
        ax1.text(xi + w/2, ox + ox * 0.01, f"{ox:,.0f}", ha="center", va="bottom",
                 fontsize=8, color=C_TEXT, **inter_props)

    ax1.set_xticks(x)
    ax1.set_xticklabels([f"Маршрут {l}" for l in labels], **inter_props)
    ax1.set_ylabel("руб.", color=C_MUTED, **inter_props)
    ax1.set_title("Выручка vs VarEx", fontsize=13, color=C_TEXT, pad=12, **inter_props)
    ax1.legend(frameon=False, fontsize=9)
    ax1.grid(axis="y", color=C_GRID, linewidth=0.7)
    ax1.set_axisbelow(True)
    for spine in ax1.spines.values():
        spine.set_visible(False)
    ax1.tick_params(colors=C_MUTED, length=0)

    # ── ROW 1: Опер. результат (горизонтальные бары) ───────────────────
    ax2 = fig.add_subplot(gs[1, 1], facecolor=C_BG)
    margins = [route_nums[l]["marginal_profit"] for l in labels]
    # Подсвечиваем убыточные направления красным, прибыльные — синим
    bar_colors = [C_MARGIN if m >= 0 else C_VAREX for m in margins]
    bars2 = ax2.barh([f"Маршрут {l}" for l in labels], margins, color=bar_colors, alpha=0.85, edgecolor="none", height=0.5)
    
    # Текстовые подписи сбоку от горизонтальных баров
    for bar, val in zip(bars2, margins):
        offset = val * 0.02 if val >= 0 else val * 0.02
        ax2.text(val + offset, bar.get_y() + bar.get_height() / 2,
                 f"{val:,.0f} ₽", ha="left" if val >= 0 else "right",
                 va="center", fontsize=9, color=C_TEXT, **inter_props)
    ax2.axvline(0, color=C_MUTED, linewidth=0.8)
    ax2.set_title("Опер. результат", fontsize=13, color=C_TEXT, pad=12, **inter_props)
    ax2.grid(axis="x", color=C_GRID, linewidth=0.7)
    ax2.set_axisbelow(True)
    for spine in ax2.spines.values():
        spine.set_visible(False)
    ax2.tick_params(colors=C_MUTED, length=0)

    # ── ROW 2: Удельная прибыль на 1 км пробега ──────────────────────────────
    ax3 = fig.add_subplot(gs[2, 0], facecolor=C_BG)
    ppkm = [route_nums[l]["profit_per_km"] for l in labels]
    colors3 = [C_PROFKM if v >= 0 else C_VAREX for v in ppkm]
    bars3 = ax3.bar(x, ppkm, 0.5, color=colors3, alpha=0.85, edgecolor="none")
    for xi, val in enumerate(ppkm):
        ax3.text(xi, val + abs(val) * 0.02, f"{val:,.2f}", ha="center", va="bottom",
                 fontsize=9, color=C_TEXT, **inter_props)
    
    # Рисуем пунктирную линию среднего значения по всем маршрутам
    avg_ppkm = g.get("profit_per_km", 0)
    ax3.axhline(avg_ppkm, color=C_MUTED, linestyle="--", linewidth=1, alpha=0.7)
    ax3.text(len(labels) - 0.5, avg_ppkm, f"  ср. {avg_ppkm:,.2f}", fontsize=8,
             va="bottom", color=C_MUTED, **inter_props)
    ax3.set_xticks(x)
    ax3.set_xticklabels([f"Маршрут {l}" for l in labels], **inter_props)
    ax3.set_ylabel("₽/км", color=C_MUTED, **inter_props)
    ax3.set_title("Удельный опер. результат на км", fontsize=13, color=C_TEXT, pad=12, **inter_props)
    ax3.axhline(0, color=C_MUTED, linewidth=0.8)
    ax3.grid(axis="y", color=C_GRID, linewidth=0.7)
    ax3.set_axisbelow(True)
    for spine in ax3.spines.values():
        spine.set_visible(False)
    ax3.tick_params(colors=C_MUTED, length=0)

    # ── ROW 2: CMR (%) ────────────────────────────────
    ax4 = fig.add_subplot(gs[2, 1], facecolor=C_BG)
    cmrs = [route_nums[l]["cmr_pct"] for l in labels]
    colors4 = [C_CMR if v >= 0 else C_VAREX for v in cmrs]
    bars4 = ax4.bar(x, cmrs, 0.5, color=colors4, alpha=0.85, edgecolor="none")
    for xi, val in enumerate(cmrs):
        ax4.text(xi, val + abs(val) * 0.02, f"{val:.1f}%", ha="center", va="bottom",
                 fontsize=9, color=C_TEXT, **inter_props)
    
    # Рисуем линию средней рентабельности по всей сети
    avg_cmr = g.get("cmr_pct", 0)
    ax4.axhline(avg_cmr, color=C_MUTED, linestyle="--", linewidth=1, alpha=0.7)
    ax4.text(len(labels) - 0.5, avg_cmr, f"  ср. {avg_cmr:.1f}%", fontsize=8,
             va="bottom", color=C_MUTED, **inter_props)
    ax4.set_xticks(x)
    ax4.set_xticklabels([f"Маршрут {l}" for l in labels], **inter_props)
    ax4.set_ylabel("%", color=C_MUTED, **inter_props)
    ax4.set_title("CMR (%)", fontsize=13, color=C_TEXT, pad=12, **inter_props)
    ax4.axhline(0, color=C_MUTED, linewidth=0.8)
    ax4.grid(axis="y", color=C_GRID, linewidth=0.7)
    ax4.set_axisbelow(True)
    for spine in ax4.spines.values():
        spine.set_visible(False)
    ax4.tick_params(colors=C_MUTED, length=0)

    plt.savefig(output_file, dpi=_DPI, bbox_inches="tight", facecolor=C_BG)
    plt.close()
    log.info(f"Финансовый дашборд сохранён: {output_file.name}")
    return output_file