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
    def _hourly_means(history: List[Tuple[float, int]]) -> List[float]:
        buckets: List[List[float]] = [[] for _ in range(24)]
        for time, count in history:
            buckets[int(time // 60) % 24].append(count)
        return [float(np.mean(b)) if b else 0.0 for b in buckets]

    def _build_hourly_util_data(self, trams: Dict) -> Dict[int, List[float]]:
        result = {}
        for tram_id, tram in trams.items():
            stop_log = getattr(getattr(tram, "stats", None), "stop_log", None) or []
            if not stop_log:
                continue
            buckets: List[List[float]] = [[] for _ in range(24)]
            for event in stop_log:
                hour = int(event["time"] // 60) % 24
                buckets[hour].append(event["utilization_after"])
            result[tram_id] = [
                float(np.mean(b)) if b else 0.0 for b in buckets
            ]
        return result

    @staticmethod
    def _collect_deviations(trams: Dict) -> List[dict]:
        """Собирает все schedule_deviations из всех трамваев в один плоский список."""
        result = []
        for tram in trams.values():
            devs = getattr(getattr(tram, "stats", None), "schedule_deviations", None) or []
            result.extend(devs)
        return result

    # ── Графики остановок ─────────────────────────────────────────────────────

    def plot_waiting_passengers(
        self, output_file: str | Path = "waiting_passengers.png"
    ) -> Path:
        output_file = Path(output_file)
        log.info("Создание графика динамики очередей...")

        ncols = 3
        nrows = (self.stop_number + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols, figsize=(18, nrows * 4))
        fig.suptitle(
            self._title("Количество ожидающих пассажиров на остановках"),
            fontsize=16, fontweight="bold"
        )

        if self.stop_number == 1:
            axes_flat = [axes]
        elif nrows == 1:
            axes_flat = list(axes)
        else:
            axes_flat = list(axes.flatten())

        for idx, stop_id in enumerate(self.stop_ids):
            stop = self.stops[stop_id]
            ax = axes_flat[idx]
            local_num = self.stop_labels[stop_id]

            if not stop.waiting_history:
                ax.text(0.5, 0.5, "Нет данных",
                        ha="center", va="center", transform=ax.transAxes)
                ax.set_title(f"Остановка {local_num}")
                continue

            times  = [t / 60 for t, _ in stop.waiting_history]
            counts = [c for _, c in stop.waiting_history]

            ax.plot(times, counts, linewidth=1.5, color="steelblue", alpha=0.7)
            ax.fill_between(times, counts, alpha=0.3, color="lightblue")
            ax.set_title(f"Остановка {local_num}", fontweight="bold")
            ax.set_xlabel("Время (часы)", fontsize=9)
            ax.set_ylabel("Количество пассажиров", fontsize=9)
            ax.grid(True, alpha=0.3, linestyle="--")
            ax.set_xlim(0, self.simulation_hours)

            max_w = max(counts)
            avg_w = sum(counts) / len(counts)
            ax.text(
                0.98, 0.95, f"Max: {max_w}\nAvg: {avg_w:.0f}",
                transform=ax.transAxes, fontsize=8,
                va="top", ha="right",
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
            )

        for idx in range(self.stop_number, len(axes_flat)):
            fig.delaxes(axes_flat[idx])

        plt.tight_layout()
        plt.savefig(output_file, dpi=_DPI, bbox_inches="tight")
        plt.close()
        log.info(f"График сохранён: {output_file.name}")
        return output_file

    def plot_waiting_by_hour(
        self,
        output_file: str | Path = "waiting_by_hour.png",
        plot_all: bool = False,
    ) -> Path:
        output_file = Path(output_file)
        log.info("Создание графика ожидания по часам...")

        hours = _OP_HOURS
        stop_data = {
            stop_id: self._hourly_means(self.stops[stop_id].waiting_history)[HOUR_START:]
            for stop_id in self.stop_ids
        }

        selected = self.stop_ids if plot_all else self.stop_ids[::2]
        colors = plt.cm.tab20(np.linspace(0, 1, max(len(selected), 1)))

        fig, ax = plt.subplots(figsize=(14, 8))
        for idx, stop_id in enumerate(selected):
            local_num = self.stop_labels[stop_id]
            ax.plot(
                hours, stop_data[stop_id],
                marker="o", linewidth=2, markersize=4,
                color=colors[idx], label=f"Остановка {local_num}",
            )

        ax.set_title(
            self._title("Среднее количество ожидающих пассажиров по часам"),
            fontsize=14, fontweight="bold",
        )
        ax.set_xlabel("Час дня", fontsize=12)
        ax.set_ylabel("Среднее количество пассажиров", fontsize=12)
        ax.set_xticks(hours)
        ax.grid(True, alpha=0.3, linestyle="--")
        ax.legend(loc="best", fontsize=9, ncol=2)

        self._add_peak_spans(ax)

        all_values = [v for d in stop_data.values() for v in d]
        y_label = max(all_values) * 0.92 if all_values and max(all_values) > 0 else 1.0
        self._add_peak_labels(ax, y_label)

        plt.tight_layout()
        plt.savefig(output_file, dpi=_DPI, bbox_inches="tight")
        plt.close()
        log.info(f"График сохранён: {output_file.name}")
        return output_file

    def plot_heatmap(
        self, output_file: str | Path = "waiting_heatmap.png"
    ) -> Path:
        output_file = Path(output_file)
        log.info("Создание тепловой карты...")

        data = np.zeros((self.stop_number, len(_OP_HOURS)))
        for idx, stop_id in enumerate(self.stop_ids):
            full = self._hourly_means(self.stops[stop_id].waiting_history)
            data[idx] = full[HOUR_START:]

        fig, ax = plt.subplots(figsize=(14, max(8, self.stop_number * 0.4)))
        im = ax.imshow(data, cmap="YlOrRd", aspect="auto", interpolation="nearest")

        ax.set_xticks(range(len(_OP_HOURS)))
        ax.set_xticklabels(_OP_HOURS)
        ax.set_yticks(range(self.stop_number))
        ax.set_yticklabels([f"Ост. {self.stop_labels[sid]}" for sid in self.stop_ids])
        ax.set_title(
            self._title("Тепловая карта: количество ожидающих пассажиров"),
            fontsize=14, fontweight="bold",
        )
        ax.set_xlabel("Час дня", fontsize=12)
        ax.set_ylabel("Остановка", fontsize=12)

        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label("Количество пассажиров", fontsize=10)

        ax.set_xticks(np.arange(24) - 0.5, minor=True)
        ax.set_yticks(np.arange(self.stop_number) - 0.5, minor=True)
        ax.grid(which="minor", color="white", linestyle="-", linewidth=0.5)

        plt.tight_layout()
        plt.savefig(output_file, dpi=_DPI, bbox_inches="tight")
        plt.close()
        log.info(f"Тепловая карта сохранена: {output_file.name}")
        return output_file

    # ── Графики трамваев ──────────────────────────────────────────────────────

    def plot_utilization(
        self, trams: Dict, output_file: str | Path = "tram_utilization.png"
    ) -> Path:
        output_file = Path(output_file)
        log.info("Создание графика загруженности трамваев...")

        active = sorted(
            [t for t in trams.values() if t.stats.total_trips > 0],
            key=lambda t: t.tram_id,
        )
        tram_ids = [t.tram_id for t in active]
        avg_utils = [
            float(np.mean(t.stats.utilization_history) * 100)
            if t.stats.utilization_history else 0.0
            for t in active
        ]

        def _bar_color(u: float) -> str:
            if u >= self.overload * 100:   return "red"
            if u >= self.comfort_high * 100: return "orange"
            if u >= self.comfort_low * 100:  return "green"
            return "steelblue"

        colors = [_bar_color(u) for u in avg_utils]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

        ax1.bar(tram_ids, avg_utils, color=colors, alpha=0.75, edgecolor="black")
        ax1.axhline(self.target_util * 100, color="blue", linestyle="--",
                    linewidth=2, label=f"Целевая ({self.target_util:.0%})")
        ax1.axhline(self.comfort_low * 100, color="green", linestyle=":",
                    alpha=0.5, label="Комфортная зона (нижняя)")
        ax1.axhline(self.overload * 100, color="red", linestyle=":",
                    alpha=0.5, label="Порог перегруза")
        ax1.set_title(self._title("Средняя загруженность трамваев"), fontweight="bold")
        ax1.set_xlabel("ID трамвая")
        ax1.set_ylabel("Загруженность (%)")
        ax1.set_ylim(0, 110)
        ax1.legend()
        ax1.grid(True, alpha=0.3, axis="y")

        trips      = [t.stats.total_trips for t in active]
        passengers = [t.stats.passengers_served for t in active]

        ax2.bar(tram_ids, trips, alpha=0.7, label="Рейсы", color="steelblue")
        ax2_twin = ax2.twinx()
        ax2_twin.plot(tram_ids, passengers, color="red", marker="o",
                      linewidth=2, markersize=6, label="Пассажиры")
        ax2.set_title(self._title("Рейсы и обслуженные пассажиры"), fontweight="bold")
        ax2.set_xlabel("ID трамвая")
        ax2.set_ylabel("Количество рейсов", color="steelblue")
        ax2_twin.set_ylabel("Обслужено пассажиров", color="red")
        ax2.legend(loc="upper left")
        ax2_twin.legend(loc="upper right")
        ax2.grid(True, alpha=0.3, axis="y")

        plt.tight_layout()
        plt.savefig(output_file, dpi=_DPI, bbox_inches="tight")
        plt.close()
        log.info(f"График загруженности сохранён: {output_file.name}")
        return output_file

    def plot_tram_utilization_by_hour(
        self,
        trams: Dict,
        output_file: str | Path = "tram_utilization_by_hour.png",
    ) -> Path:
        output_file = Path(output_file)
        log.info("Создание графика загруженности по часам...")

        hourly_data = self._build_hourly_util_data(trams)
        if not hourly_data:
            log.warning("Нет данных для графика загруженности по часам — пропускаем")
            return output_file

        hours = _OP_HOURS
        avg_util = []
        for hour in hours:
            vals = [hourly_data[tid][hour] for tid in hourly_data if hourly_data[tid][hour] > 0]
            avg_util.append(float(np.mean(vals)) if vals else 0.0)

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

        ax1.plot(hours, avg_util, linewidth=3, color="steelblue",
                 marker="o", markersize=6, label="Средняя загруженность")
        ax1.fill_between(hours, avg_util, alpha=0.3, color="lightblue")
        ax1.axhline(self.target_util * 100, color="green", linestyle="--",
                    linewidth=2, label=f"Целевая ({self.target_util:.0%})", alpha=0.7)
        ax1.axhspan(self.comfort_low * 100, self.comfort_high * 100,
                    alpha=0.1, color="green", label="Комфортная зона")
        self._add_peak_spans(ax1)
        ax1.set_title(self._title("Средняя загруженность трамваев по часам суток"),
                      fontsize=14, fontweight="bold")
        ax1.set_xlabel("Час дня", fontsize=12)
        ax1.set_ylabel("Загруженность (%)", fontsize=12)
        ax1.set_xticks(hours)
        ax1.set_ylim(0, 100)
        ax1.grid(True, alpha=0.3, linestyle="--")
        ax1.legend(loc="best", fontsize=10)

        y_label = max(avg_util) * 0.90 if max(avg_util) > 0 else 5.0
        self._add_peak_labels(ax1, y_label)

        selected = sorted(hourly_data.keys())[::2]
        colors = plt.cm.tab10(np.linspace(0, 1, max(len(selected), 1)))
        for idx, tram_id in enumerate(selected):
            ax2.plot(hours, hourly_data[tram_id], linewidth=1.5, marker="o",
                     markersize=3, color=colors[idx], alpha=0.7, label=f"Трамвай #{tram_id}")
        ax2.plot(hours, avg_util, linewidth=2.5, color="black",
                 linestyle="--", label="Средняя", alpha=0.5)
        self._add_peak_spans(ax2)
        ax2.set_title(self._title("Загруженность отдельных трамваев по часам"),
                      fontsize=14, fontweight="bold")
        ax2.set_xlabel("Час дня", fontsize=12)
        ax2.set_ylabel("Загруженность (%)", fontsize=12)
        ax2.set_xticks(hours)
        ax2.set_ylim(0, 100)
        ax2.grid(True, alpha=0.3, linestyle="--")
        ax2.legend(loc="best", fontsize=9, ncol=2)

        plt.tight_layout()
        plt.savefig(output_file, dpi=_DPI, bbox_inches="tight")
        plt.close()
        log.info(f"График загруженности по часам сохранён: {output_file.name}")
        return output_file

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

        tasks = [
            ("waiting_passengers.png",  lambda p: self.plot_waiting_passengers(p)),
            ("waiting_by_hour.png",     lambda p: self.plot_waiting_by_hour(p)),
            ("waiting_heatmap.png",     lambda p: self.plot_heatmap(p)),
        ]
        if trams:
            tasks += [
                ("tram_utilization.png",         lambda p: self.plot_utilization(trams, p)),
                ("tram_utilization_by_hour.png", lambda p: self.plot_tram_utilization_by_hour(trams, p)),
                # ── три новых графика отклонений ──────────────────────────────
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