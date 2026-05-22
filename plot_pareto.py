# plot_pareto.py
"""
Модуль визуализации результатов многокритериальной оптимизации (Парето-фронта) NSGA-II.

В рамках задачи оптимизации мы ищем баланс между двумя противоречивыми целями:
1. Качество соблюдения интервалов (headway_mae_min, в минутах, меньше — лучше).
2. Общий полученный доход (total_revenue, в рублях, больше — лучше).

Модуль строит три информативных графика для лица, принимающего решения:
1. Зависимость дохода от точности интервалов (с аннотациями экстремальных решений).
2. Детальный столбчатый график распределения парка трамваев по маршрутам для Топ-10 лучших решений.
3. Взаимосвязь суммарного размера парка и итогового дохода с цветовой индикацией ошибок интервалов.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sys
import os


def plot_pareto(csv_path: str, out_dir: str = None):
    """
    Считывает CSV-файл с решениями Парето-фронта и генерирует три графика.

    :param csv_path: Путь к файлу pareto_front.csv, сгенерированному оптимизатором.
    :param out_dir: Папка для сохранения PNG-изображений графиков. Если не задана, используется папка файла CSV.
    """
    df = pd.read_csv(csv_path)
    out_dir = out_dir or os.path.dirname(csv_path) or "."

    # Нормализуем доход (pymoo минимизирует, поэтому в исходных расчетах выручка могла быть отрицательной)
    if "total_revenue" in df.columns:
        df["total_revenue"] = df["total_revenue"].abs()

    # Суммарное число трамваев по каждому решению для цветовой нормализации
    total_trams = df[["n_20", "n_48", "n_55"]].sum(axis=1).values
    norm  = plt.Normalize(total_trams.min(), total_trams.max())
    cmap  = "viridis"  # Красивая и понятная палитра

    def _scatter(ax, x_col, y_col, x_label, y_label):
        """Вспомогательный метод для построения scatter-графика."""
        sc = ax.scatter(
            df[x_col], df[y_col],
            c=total_trams, cmap=cmap, norm=norm,
            s=80, edgecolors="k", linewidths=0.4, alpha=0.9,
        )
        ax.set_xlabel(x_label, fontsize=10)
        ax.set_ylabel(y_label, fontsize=10)
        ax.grid(True, alpha=0.3, linestyle="--")
        return sc

    def _annotate_extremes(ax, x_col, y_col):
        """Вспомогательный метод для добавления стрелок и текстовых аннотаций к экстремальным решениям."""
        candidates = {
            "макс доход":  df["total_revenue"].idxmax()   if "total_revenue" in df.columns else None,
            "мин mae":     df["headway_mae_min"].idxmin()  if "headway_mae_min" in df.columns else None,
        }
        for label, idx in candidates.items():
            if idx is None:
                continue
            # Формируем красивую подпись с распределением парка: [n_20, n_48, n_55]
            ax.annotate(
                f'{label}\n[{int(df.loc[idx,"n_20"])},{int(df.loc[idx,"n_48"])},{int(df.loc[idx,"n_55"])}]',
                xy=(df.loc[idx, x_col], df.loc[idx, y_col]),
                xytext=(10, 10), textcoords="offset points",
                fontsize=7, color="darkred",
                arrowprops=dict(arrowstyle="->", color="darkred", lw=0.8),
            )

    created = []

    # ── 1. Основной 2D график: Выручка vs Точность интервалов (MAE) ──────────
    fig, ax = plt.subplots(figsize=(10, 7))
    sc = _scatter(ax, "total_revenue", "headway_mae_min",
                  "Доход (руб.) →",
                  "← MAE интервалов (мин) — меньше лучше")
    _annotate_extremes(ax, "total_revenue", "headway_mae_min")

    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label("Суммарный парк (трамваев)", fontsize=9)

    ax.set_title(
        "Pareto-фронт NSGA-II — Доход vs Качество интервалов\n"
        "Маршруты 20, 48, 55",
        fontsize=12, fontweight="bold",
    )
    plt.tight_layout()
    out_path = os.path.join(out_dir, "pareto_revenue_vs_mae.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Сохранён: {out_path}")
    created.append(out_path)

    # ── 2. Столбчатый график: распределение парка для Топ-10 решений ──────────
    top_n = min(10, len(df))
    # Сортируем решения по выручке и берем первые top_n
    df_sorted = df.sort_values("total_revenue", ascending=False).head(top_n).reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(12, 6))
    x_pos = np.arange(top_n)
    width = 0.25

    # Строим три столбца для каждого решения (по числу трамваев на каждом маршруте)
    bars_20 = ax.bar(x_pos - width, df_sorted["n_20"], width, label="Маршрут 20", color="#2196F3", alpha=0.85)
    bars_48 = ax.bar(x_pos,         df_sorted["n_48"], width, label="Маршрут 48", color="#4CAF50", alpha=0.85)
    bars_55 = ax.bar(x_pos + width, df_sorted["n_55"], width, label="Маршрут 55", color="#FF9800", alpha=0.85)

    # Добавляем текстовые подписи (Доход / MAE) над каждой группой столбцов
    for i in range(top_n):
        total = df_sorted.loc[i, "total_revenue"]
        mae   = df_sorted.loc[i, "headway_mae_min"]
        ax.text(i, df_sorted.loc[i, ["n_20", "n_48", "n_55"]].max() + 0.5,
                f"{total/1000:.0f}к\n{mae:.1f}м",
                ha="center", fontsize=7, color="darkred")

    ax.set_xlabel("Решение (ранжировано по убыванию дохода)", fontsize=11)
    ax.set_ylabel("Количество трамваев", fontsize=11)
    ax.set_title(
        f"Топ-{top_n} решений Pareto: распределение парка по маршрутам",
        fontsize=12, fontweight="bold",
    )
    ax.set_xticks(x_pos)
    ax.set_xticklabels([f"#{i+1}" for i in range(top_n)])
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis="y", linestyle="--")

    plt.tight_layout()
    out_path = os.path.join(out_dir, "pareto_fleet_distribution.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Сохранён: {out_path}")
    created.append(out_path)

    # ── 3. Размер парка vs Доход с цветовой индикацией точности ─────────────
    if "total_trams" in df.columns:
        fig, ax = plt.subplots(figsize=(9, 6))
        # Цветовая шкала: RdYlGn_r (красный -> желтый -> зеленый, реверсивный). 
        # Зеленый — маленькая ошибка MAE (отлично), красный — большая ошибка.
        sc = ax.scatter(
            df["total_trams"], df["total_revenue"],
            c=df["headway_mae_min"], cmap="RdYlGn_r",
            s=80, edgecolors="k", linewidths=0.4, alpha=0.9,
        )
        ax.set_xlabel("Суммарный парк (трамваев) →", fontsize=10)
        ax.set_ylabel("Доход (руб.) →", fontsize=10)
        ax.set_title(
            "Размер парка vs Доход (цвет = MAE интервалов)",
            fontsize=12, fontweight="bold",
        )
        cbar = fig.colorbar(sc, ax=ax)
        cbar.set_label("MAE интервалов (мин)", fontsize=9)
        ax.grid(True, alpha=0.3, linestyle="--")

        plt.tight_layout()
        out_path = os.path.join(out_dir, "pareto_fleet_vs_revenue.png")
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Сохранён: {out_path}")
        created.append(out_path)

    print(f"\nВсего графиков Парето сгенерировано: {len(created)}")
    return created


# ─── Точка запуска из консоли ────────────────────────────────────────────────

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Использование: python plot_pareto.py <путь до pareto_front.csv> [out_dir]")
        sys.exit(1)

    out_dir = sys.argv[2] if len(sys.argv) > 2 else None
    plot_pareto(csv_path=sys.argv[1], out_dir=out_dir)
