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
    """Отрисовывает сводный финансовый дашборд (Выручка, OpEx, Маржа, ROS)."""
    import matplotlib.font_manager as fm

    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    log.info("Создание финансового дашборда...")

    routes_stats = stats.get("routes", {})
    global_stats = stats.get("global", {})

    # Группируем fwd+bwd в один маршрут (по номеру)
    route_nums: Dict[str, dict] = {}
    for rid, rs in routes_stats.items():
        num = rid.split("_")[0]
        if num not in route_nums:
            route_nums[num] = {
                "revenue": 0, "opex": 0, "marginal_profit": 0,
                "tram_km": 0, "total_trips": 0,
            }
        route_nums[num]["revenue"] += rs.get("revenue", 0)
        route_nums[num]["opex"] += rs.get("opex", 0)
        route_nums[num]["marginal_profit"] += rs.get("marginal_profit", 0)
        route_nums[num]["tram_km"] += rs.get("tram_km", 0)
        route_nums[num]["total_trips"] += rs.get("total_trips", 0)

    for num, d in route_nums.items():
        d["profit_per_km"] = d["marginal_profit"] / d["tram_km"] if d["tram_km"] > 0 else 0
        d["ros_pct"] = (d["marginal_profit"] / d["revenue"] * 100) if d["revenue"] > 0 else 0

    labels = sorted(route_nums.keys())
    if not labels:
        log.warning("Нет данных о маршрутах для дашборда.")
        return output_file

    # ── Шрифт Inter ──────────────────────────────────────────────────────────
    inter_props = {}
    for fp in fm.findSystemFonts():
        try:
            name = fm.FontProperties(fname=fp).get_name()
            if "inter" in name.lower():
                inter_props = {"fontproperties": fm.FontProperties(fname=fp)}
                break
        except Exception:
            continue

    # ── Цветовая палитра ─────────────────────────────────────────────────────
    C_BG       = "#FFFFFF"
    C_TEXT     = "#2D2D2D"
    C_MUTED    = "#8C8C8C"
    C_GRID     = "#E8E8E8"
    C_REVENUE  = "#34A853"
    C_OPEX     = "#EA4335"
    C_MARGIN   = "#4285F4"
    C_ROS      = "#FBBC05"
    C_PROFKM   = "#7B61FF"
    C_BAR_REV  = "#34A853"
    C_BAR_OPEX = "#EA4335"

    # ── Layout: 3 строки × 2 колонки ─────────────────────────────────────────
    fig = plt.figure(figsize=(16, 14), facecolor=C_BG)
    gs = fig.add_gridspec(
        3, 2,
        height_ratios=[0.22, 1, 1],
        hspace=0.35, wspace=0.30,
        left=0.07, right=0.95, top=0.94, bottom=0.05,
    )

    # ── ROW 0: KPI-карточки ──────────────────────────────────────────────────
    ax_kpi = fig.add_subplot(gs[0, :])
    ax_kpi.set_xlim(0, 1)
    ax_kpi.set_ylim(0, 1)
    ax_kpi.axis("off")

    g = global_stats
    kpis = [
        ("Выручка",            g.get("total_revenue", 0),       "₽", C_REVENUE),
        ("OpEx",               g.get("opex", 0),                "₽", C_OPEX),
        ("Марж. прибыль",      g.get("marginal_profit", 0),     "₽", C_MARGIN),
        ("Прибыль / км",       g.get("profit_per_km", 0),       "₽/км", C_PROFKM),
        ("ROS",                g.get("ros_pct", 0),             "%", C_ROS),
    ]

    card_w = 1.0 / len(kpis)
    for i, (title, value, unit, color) in enumerate(kpis):
        cx = card_w * i + card_w / 2
        # Заголовок
        ax_kpi.text(cx, 0.82, title.upper(), fontsize=10, ha="center", va="center",
                    color=C_MUTED, fontweight="normal", **inter_props)
        # Значение
        if unit == "%":
            val_str = f"{value:+.1f}{unit}"
        elif abs(value) >= 1_000_000:
            val_str = f"{value/1_000_000:,.2f} М{unit}"
        elif abs(value) >= 1_000:
            val_str = f"{value:,.0f} {unit}"
        else:
            val_str = f"{value:,.2f} {unit}"
        ax_kpi.text(cx, 0.38, val_str, fontsize=22, ha="center", va="center",
                    color=color, fontweight="bold", **inter_props)
        # Разделитель
        if i < len(kpis) - 1:
            ax_kpi.axvline(card_w * (i + 1), color=C_GRID, linewidth=1, ymin=0.15, ymax=0.85)

    # Рамка
    for spine in ["top", "bottom", "left", "right"]:
        ax_kpi.spines[spine].set_visible(False)
    rect = matplotlib.patches.FancyBboxPatch(
        (0.005, 0.05), 0.99, 0.90, boxstyle="round,pad=0.02",
        facecolor="#F9F9F9", edgecolor=C_GRID, linewidth=1.5,
        transform=ax_kpi.transAxes, zorder=-1,
    )
    ax_kpi.add_patch(rect)

    # ── ROW 1: Выручка vs OpEx (grouped bar) ────────────────────────────────
    ax1 = fig.add_subplot(gs[1, 0], facecolor=C_BG)
    x = np.arange(len(labels))
    w = 0.35
    revs = [route_nums[l]["revenue"] for l in labels]
    opexs = [route_nums[l]["opex"] for l in labels]

    ax1.bar(x - w/2, revs,  w, color=C_BAR_REV,  alpha=0.85, label="Выручка", edgecolor="none")
    ax1.bar(x + w/2, opexs, w, color=C_BAR_OPEX, alpha=0.85, label="OpEx",    edgecolor="none")

    for xi, (rv, ox) in enumerate(zip(revs, opexs)):
        ax1.text(xi - w/2, rv + rv * 0.01, f"{rv:,.0f}", ha="center", va="bottom",
                 fontsize=8, color=C_TEXT, **inter_props)
        ax1.text(xi + w/2, ox + ox * 0.01, f"{ox:,.0f}", ha="center", va="bottom",
                 fontsize=8, color=C_TEXT, **inter_props)

    ax1.set_xticks(x)
    ax1.set_xticklabels([f"Маршрут {l}" for l in labels], **inter_props)
    ax1.set_ylabel("руб.", color=C_MUTED, **inter_props)
    ax1.set_title("Выручка vs OpEx", fontsize=13, color=C_TEXT, pad=12, **inter_props)
    ax1.legend(frameon=False, fontsize=9)
    ax1.grid(axis="y", color=C_GRID, linewidth=0.7)
    ax1.set_axisbelow(True)
    for spine in ax1.spines.values():
        spine.set_visible(False)
    ax1.tick_params(colors=C_MUTED, length=0)

    # ── ROW 1: Маржинальная прибыль (горизонтальный bar) ─────────────────────
    ax2 = fig.add_subplot(gs[1, 1], facecolor=C_BG)
    margins = [route_nums[l]["marginal_profit"] for l in labels]
    bar_colors = [C_MARGIN if m >= 0 else C_OPEX for m in margins]
    bars2 = ax2.barh([f"Маршрут {l}" for l in labels], margins, color=bar_colors, alpha=0.85, edgecolor="none", height=0.5)
    for bar, val in zip(bars2, margins):
        offset = val * 0.02 if val >= 0 else val * 0.02
        ax2.text(val + offset, bar.get_y() + bar.get_height() / 2,
                 f"{val:,.0f} ₽", ha="left" if val >= 0 else "right",
                 va="center", fontsize=9, color=C_TEXT, **inter_props)
    ax2.axvline(0, color=C_MUTED, linewidth=0.8)
    ax2.set_title("Маржинальная прибыль", fontsize=13, color=C_TEXT, pad=12, **inter_props)
    ax2.grid(axis="x", color=C_GRID, linewidth=0.7)
    ax2.set_axisbelow(True)
    for spine in ax2.spines.values():
        spine.set_visible(False)
    ax2.tick_params(colors=C_MUTED, length=0)

    # ── ROW 2: Удельная прибыль на км ────────────────────────────────────────
    ax3 = fig.add_subplot(gs[2, 0], facecolor=C_BG)
    ppkm = [route_nums[l]["profit_per_km"] for l in labels]
    colors3 = [C_PROFKM if v >= 0 else C_OPEX for v in ppkm]
    bars3 = ax3.bar(x, ppkm, 0.5, color=colors3, alpha=0.85, edgecolor="none")
    for xi, val in enumerate(ppkm):
        ax3.text(xi, val + abs(val) * 0.02, f"{val:,.2f}", ha="center", va="bottom",
                 fontsize=9, color=C_TEXT, **inter_props)
    avg_ppkm = g.get("profit_per_km", 0)
    ax3.axhline(avg_ppkm, color=C_MUTED, linestyle="--", linewidth=1, alpha=0.7)
    ax3.text(len(labels) - 0.5, avg_ppkm, f"  ср. {avg_ppkm:,.2f}", fontsize=8,
             va="bottom", color=C_MUTED, **inter_props)
    ax3.set_xticks(x)
    ax3.set_xticklabels([f"Маршрут {l}" for l in labels], **inter_props)
    ax3.set_ylabel("₽/км", color=C_MUTED, **inter_props)
    ax3.set_title("Удельная маржинальная прибыль на км", fontsize=13, color=C_TEXT, pad=12, **inter_props)
    ax3.axhline(0, color=C_MUTED, linewidth=0.8)
    ax3.grid(axis="y", color=C_GRID, linewidth=0.7)
    ax3.set_axisbelow(True)
    for spine in ax3.spines.values():
        spine.set_visible(False)
    ax3.tick_params(colors=C_MUTED, length=0)

    # ── ROW 2: ROS (рентабельность) ──────────────────────────────────────────
    ax4 = fig.add_subplot(gs[2, 1], facecolor=C_BG)
    ross = [route_nums[l]["ros_pct"] for l in labels]
    colors4 = [C_ROS if v >= 0 else C_OPEX for v in ross]
    bars4 = ax4.bar(x, ross, 0.5, color=colors4, alpha=0.85, edgecolor="none")
    for xi, val in enumerate(ross):
        ax4.text(xi, val + abs(val) * 0.02, f"{val:.1f}%", ha="center", va="bottom",
                 fontsize=9, color=C_TEXT, **inter_props)
    avg_ros = g.get("ros_pct", 0)
    ax4.axhline(avg_ros, color=C_MUTED, linestyle="--", linewidth=1, alpha=0.7)
    ax4.text(len(labels) - 0.5, avg_ros, f"  ср. {avg_ros:.1f}%", fontsize=8,
             va="bottom", color=C_MUTED, **inter_props)
    ax4.set_xticks(x)
    ax4.set_xticklabels([f"Маршрут {l}" for l in labels], **inter_props)
    ax4.set_ylabel("%", color=C_MUTED, **inter_props)
    ax4.set_title("ROS (Рентабельность продаж)", fontsize=13, color=C_TEXT, pad=12, **inter_props)
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