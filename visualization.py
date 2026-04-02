"""
Модуль визуализации результатов симуляции трамваев.
"""

from __future__ import annotations

import logging
import traceback
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use("Agg")
plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans", "sans-serif"]
plt.rcParams["axes.unicode_minus"] = False

log = logging.getLogger(__name__)

# ─── Дефолтные константы оформления ──────────────────────────────────────────
_DEFAULT_TARGET_UTIL   = 0.75
_DEFAULT_COMFORT_LOW   = 0.60
_DEFAULT_COMFORT_HIGH  = 0.80
_DEFAULT_OVERLOAD      = 0.90
_DEFAULT_PEAK_RANGES   = [(7, 9), (17, 19)]
_DPI                   = 150
HOUR_START  = 5   # начало операционного дня
HOUR_END    = 24  # конец (0 = полночь следующего дня)
_OP_HOURS   = list(range(HOUR_START, HOUR_END))


class TramVisualization:
    """Класс для визуализации результатов симуляции."""

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
        self.stops = stops
        self.simulation_hours = simulation_hours

        self.stop_ids: List[int] = sorted(stops.keys())
        self.stop_number: int = len(self.stop_ids)

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

    # ── Вспомогательные методы ────────────────────────────────────────────────

    def _title(self, base: str) -> str:
        return f"{base} (маршрут {self.route_id})" if self.route_id else base

    def _add_peak_spans(self, ax: plt.Axes) -> None:
        colors = ["red", "orange", "red", "orange"]
        for i, (h_start, h_end) in enumerate(self.peak_ranges):
            ax.axvspan(h_start, h_end, alpha=0.10, color=colors[i % len(colors)])

    def _add_peak_labels(self, ax: plt.Axes, y: float) -> None:
        default_labels = ["Утренний\nчас пик", "Вечерний\nчас пик"]
        for i, (h_start, h_end) in enumerate(self.peak_ranges):
            label = default_labels[i] if i < len(default_labels) else f"Пик {i+1}"
            ax.text(
                (h_start + h_end) / 2, y, label,
                ha="center", fontsize=9, alpha=0.7
            )

    @staticmethod
    def _collect_deviations(trams: Dict) -> List[dict]:
        """Собирает все schedule_deviations из всех трамваев в один плоский список."""
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
        Барчарт: средняя ошибка интервалов по каждой остановке.
        Показывает где маршрут систематически нарушает интервалы.
        """
        output_file = Path(output_file)
        log.info("Создание графика ошибок интервалов по остановкам...")

        deviations = self._collect_deviations(trams)
        if not deviations:
            log.warning("Нет данных об ошибках интервалов — пропускаем")
            return output_file

        # Группируем headway_error_min по stop_id
        stop_delays: Dict[int, List[float]] = {}
        for d in deviations:
            sid = d["stop_id"]
            if d.get("headway_error_min") is not None:
                stop_delays.setdefault(sid, []).append(d["headway_error_min"])

        # Берём только остановки которые есть в маршруте, в правильном порядке
        ordered_ids = [sid for sid in self.stop_ids if sid in stop_delays]
        labels      = [str(self.stop_labels[sid]) for sid in ordered_ids]
        means       = [float(np.mean(stop_delays[sid])) for sid in ordered_ids]
        stds        = [float(np.std(stop_delays[sid]))  for sid in ordered_ids]

        colors = ["tomato" if m > 5 else "steelblue" for m in means]

        fig, ax = plt.subplots(figsize=(max(12, len(labels) * 0.6), 6))
        bars = ax.bar(labels, means, color=colors, alpha=0.8,
                      edgecolor="black", linewidth=0.5)
        ax.errorbar(labels, means, yerr=stds, fmt="none",
                    color="black", capsize=4, linewidth=1.2, alpha=0.6)

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

        # Подписи значений на барах
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
        Линейный график: средняя ошибка интервалов по часам суток.
        Показывает как загруженность дорог влияет на соблюдение интервалов.
        """
        output_file = Path(output_file)
        log.info("Создание графика ошибок интервалов по часам...")

        deviations = self._collect_deviations(trams)
        if not deviations:
            log.warning("Нет данных об ошибках интервалов — пропускаем")
            return output_file

        # Группируем по часу (из actual_time)
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

        ax.plot(hours, means, linewidth=2.5, color="steelblue",
                marker="o", markersize=6, label="Средняя ошибка интервала")
        ax.fill_between(hours,
                        np.clip(means_arr - stds_arr, 0, None),
                        means_arr + stds_arr,
                        alpha=0.2, color="steelblue", label="±σ (разброс)")

        ax.axhline(0, color="black", linewidth=1.5)
        ax.axhline(2,  color="orange", linewidth=1, linestyle="--",
                   alpha=0.8, label="2 мин (допустимо)")
        ax.axhline(5,  color="red", linewidth=1, linestyle="--",
                   alpha=0.8, label="5 мин (критично)")

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
        Тепловая карта: ось X — час дня, ось Y — остановка, цвет — средняя ошибка.
        Самый информативный график — сразу видно проблемные участки в конкретное время.
        """
        output_file = Path(output_file)
        log.info("Создание тепловой карты ошибок интервалов...")

        deviations = self._collect_deviations(trams)
        if not deviations:
            log.warning("Нет данных об ошибках интервалов — пропускаем")
            return output_file

        # Матрица: строки — остановки, столбцы — часы
        delay_matrix: Dict[int, Dict[int, List[float]]] = {
            sid: {h: [] for h in range(24)} for sid in self.stop_ids
        }
        for d in deviations:
            if d.get("headway_error_min") is not None:
                sid  = d["stop_id"]
                hour = int(d.get("actual_time", 0) // 60) % 24
                if sid in delay_matrix:
                    delay_matrix[sid][hour].append(d["headway_error_min"])

        data = np.zeros((self.stop_number, len(_OP_HOURS)))
        for idx, sid in enumerate(self.stop_ids):
            for col, h in enumerate(_OP_HOURS):
                vals = delay_matrix[sid][h]
                data[idx, col] = float(np.mean(vals)) if vals else 0.0

        # Односторонняя цветовая шкала: темно-красный = сильное нарушение интервалов
        vmax = max(data.max(), 1.0)

        fig, ax = plt.subplots(figsize=(14, max(8, self.stop_number * 0.4)))
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

        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label("Ошибка интервала (мин)",
                       fontsize=10)

        ax.set_xticks(np.arange(24) - 0.5, minor=True)
        ax.set_yticks(np.arange(self.stop_number) - 0.5, minor=True)
        ax.grid(which="minor", color="white", linestyle="-", linewidth=0.5)

        plt.tight_layout()
        plt.savefig(output_file, dpi=_DPI, bbox_inches="tight")
        plt.close()
        log.info(f"Тепловая карта отклонений сохранена: {output_file.name}")
        return output_file

    # ── Главный метод ─────────────────────────────────────────────────────────

    def create_all_plots(
        self,
        trams: Optional[Dict] = None,
        output_dir: str | Path = ".",
    ) -> List[Path]:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        log.info("\n" + "=" * 60)
        log.info("СОЗДАНИЕ ГРАФИКОВ")
        log.info(f"Папка: {output_dir.resolve()}")
        log.info("=" * 60)

        tasks = []
        if trams:
            tasks += [
                # ── графики отклонений ────────────────────────────────────────
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
    """Отрисовывает глобальный финансовый дашборд для всей сети."""
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    log.info("Создание глобального финансового дашборда...")

    routes_stats = stats.get("routes", {})
    global_stats = stats.get("global", {})
    route_ids = sorted(list(routes_stats.keys()))

    if not route_ids:
        log.warning("Нет данных о маршрутах для глобального дашборда.")
        return output_file

    revenues = [routes_stats[r].get("revenue", 0) for r in route_ids]
    passengers = [routes_stats[r].get("passengers_estimated", 0) for r in route_ids]
    tram_kms = [routes_stats[r].get("tram_km", 0) for r in route_ids]

    rev_per_km = [r / km if km > 0 else 0.0 for r, km in zip(revenues, tram_kms)]

    # Настройка Layout панели
    fig = plt.figure(figsize=(16, 10))
    # GridSpec для разделения графиков от карточек
    gs = fig.add_gridspec(2, 3, width_ratios=[1, 1, 1.2], height_ratios=[1, 1])
    fig.suptitle("Сводный Финансовый Дашборд Сети", fontsize=20, fontweight="bold", y=0.96)

    # 1. Bar chart: Revenues
    ax1 = fig.add_subplot(gs[0, 0])
    bars1 = ax1.bar(route_ids, revenues, color="#2ca02c", alpha=0.8, edgecolor="black")
    ax1.set_title("Доходы по маршрутам", fontweight="bold")
    ax1.set_ylabel("Доход (руб.)")
    ax1.grid(True, alpha=0.3, axis="y")
    # Добавление значений над барами
    for bar, val in zip(bars1, revenues):
        ax1.text(
            bar.get_x() + bar.get_width() / 2, 
            val + val * 0.02, 
            f"{val:,.0f}", 
            ha='center', va='bottom', fontsize=9
        )

    # 2. Bar chart: Passengers
    ax2 = fig.add_subplot(gs[0, 1])
    bars2 = ax2.bar(route_ids, passengers, color="#1f77b4", alpha=0.8, edgecolor="black")
    ax2.set_title("Расчетный пассажиропоток", fontweight="bold")
    ax2.set_ylabel("Количество пассажиров")
    ax2.grid(True, alpha=0.3, axis="y")
    for bar, val in zip(bars2, passengers):
        ax2.text(
            bar.get_x() + bar.get_width() / 2, 
            val + val * 0.02, 
            f"{val:,.0f}", 
            ha='center', va='bottom', fontsize=9
        )

    # 3. Bar chart: Revenue per KM
    ax3 = fig.add_subplot(gs[1, 0:2])
    bars3 = ax3.bar(route_ids, rev_per_km, color="#ff7f0e", alpha=0.8, edgecolor="black")
    ax3.set_title("Рентабельность (Доход на 1 т-км)", fontweight="bold")
    ax3.set_ylabel("Доход / т-км (руб.)")
    ax3.grid(True, alpha=0.3, axis="y")
    for bar, val in zip(bars3, rev_per_km):
        ax3.text(
            bar.get_x() + bar.get_width() / 2, 
            val + val * 0.02, 
            f"{val:,.1f}", 
            ha='center', va='bottom', fontsize=9
        )
        
    avg_rev_per_km = sum(revenues) / sum(tram_kms) if sum(tram_kms) > 0 else 0
    ax3.axhline(avg_rev_per_km, color="red", linestyle="--", alpha=0.7, label=f"Среднее: {avg_rev_per_km:,.1f}")
    ax3.legend()

    # 4. KPI Panel (Правая колонка)
    ax4 = fig.add_subplot(gs[:, 2])
    ax4.axis("off")

    tot_rev = global_stats.get("total_revenue", 0)
    tot_pax = global_stats.get("total_passengers_est", 0)
    tot_km  = global_stats.get("total_tram_km", 0)

    x_center = 0.5
    y_start = 0.82
    y_step = 0.22

    ax4.text(x_center, 0.96, "ГЛОБАЛЬНЫЕ KPI", fontsize=18, fontweight="bold", ha="center", va="center")

    def add_kpi_card(ax, y_pos, title, value, unit, color):
        ax.text(x_center, y_pos, title, fontsize=11, ha="center", va="bottom", color="grey", fontweight="bold")
        formatted_val = f"{value:,.0f}" if isinstance(value, (int, float)) and int(value) == value else f"{value:,.1f}"
        ax.text(x_center, y_pos - 0.06, formatted_val, fontsize=28, ha="center", va="center", color=color, fontweight="bold")
        if unit:
            ax.text(x_center, y_pos - 0.12, unit, fontsize=11, ha="center", va="top", color="grey")

    add_kpi_card(ax4, y_start, "ОБЩИЙ ДОХОД СЕТИ", tot_rev, "рублей", "#2ca02c")
    add_kpi_card(ax4, y_start - y_step, "ПАССАЖИРОПОТОК", tot_pax, "человек (расч.)", "#1f77b4")
    add_kpi_card(ax4, y_start - y_step * 2, "ВЫПОЛНЕННАЯ РАБОТА", tot_km, "трамвай-километров", "black")
    add_kpi_card(ax4, y_start - y_step * 3, "СРЕДНЯЯ РЕНТАБЕЛЬНОСТЬ", avg_rev_per_km, "рублей / т-км", "#ff7f0e")

    # Добавление фона для панели KPI
    bbox = matplotlib.patches.Rectangle(
        (0.1, 0.05), 0.8, 0.95, transform=ax4.transAxes, 
        color="#f8f9fa", zorder=-1, ec="lightgrey", lw=2, alpha=0.8
    )
    ax4.add_patch(bbox)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(output_file, dpi=_DPI, bbox_inches="tight")
    plt.close()
    log.info(f"Сводный дашборд сохранён: {output_file.name}")
    return output_file