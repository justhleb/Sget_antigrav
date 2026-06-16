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


def plot_pareto(csv_path: str, out_dir: str = None, n_max: int = None):
    """
    Считывает CSV-файл с решениями Парето-фронта и генерирует три графика.

    :param csv_path: Путь к файлу pareto_front.csv, сгенерированному оптимизатором.
    :param out_dir: Папка для сохранения PNG-изображений графиков. Если не задана, используется папка файла CSV.
    """
    df = pd.read_csv(csv_path)
    out_dir = out_dir or os.path.dirname(csv_path) or "."

    # Проверяем наличие файла со всеми рассмотренными вариантами
    df_all = None
    all_csv_path = os.path.join(os.path.dirname(csv_path), "all_evaluated.csv")
    if not os.path.exists(all_csv_path) and out_dir:
        all_csv_path = os.path.join(out_dir, "all_evaluated.csv")
        
    if os.path.exists(all_csv_path):
        try:
            df_all = pd.read_csv(all_csv_path)
        except Exception as e:
            print(f"Не удалось загрузить all_evaluated.csv: {e}")

    # Определяем ключевую финансовую метрику
    if "marginal_profit" in df.columns:
        fin_col = "marginal_profit"
        fin_label = "Опер. результат (млн руб.)"
        fin_short_label = "Опер. результат"
        fin_label_short_ru = "опер. результат"
        fin_label_short_ru_genitive = "опер. результата"
        is_mp = True
    else:
        fin_col = "total_revenue"
        fin_label = "Доход (руб.)"
        fin_short_label = "Доход"
        fin_label_short_ru = "доход"
        fin_label_short_ru_genitive = "дохода"
        is_mp = False

    # Нормализуем значения финансового показателя (pymoo минимизирует отрицательные значения)
    if fin_col in df.columns:
        df[fin_col] = df[fin_col].abs()
        if is_mp:
            df[fin_col] = df[fin_col] / 1e6

    if df_all is not None and fin_col in df_all.columns:
        df_all[fin_col] = df_all[fin_col].abs()
        if is_mp:
            df_all[fin_col] = df_all[fin_col] / 1e6

    # Суммарное число трамваев по каждому решению для цветовой нормализации
    total_trams = df[["n_20", "n_48", "n_55"]].sum(axis=1).values
    
    # Определяем максимум для цветовой шкалы на основе n_max
    if n_max is None:
        if df_all is not None and "total_trams" in df_all.columns:
            n_max = int(df_all["total_trams"].max())
        else:
            n_max = int(total_trams.max())
            
    v_min = total_trams.min()
    v_max = n_max
    if v_min >= v_max:
        v_min = max(0, v_max - 10)
        
    norm  = plt.Normalize(v_min, v_max)
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
            f"макс {fin_label_short_ru}":  df[fin_col].idxmax() if fin_col in df.columns else None,
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

    # ── 1. Основной 2D график: Прибыль/Выручка vs Точность интервалов (MAE) ──────────
    fig, ax = plt.subplots(figsize=(10, 7))
    sc = _scatter(ax, fin_col, "headway_mae_min",
                  f"{fin_label} →",
                  "← MAE интервалов (мин) — меньше лучше")
    _annotate_extremes(ax, fin_col, "headway_mae_min")

    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label("Суммарный парк (трамваев)", fontsize=9)

    ax.set_title(
        f"Pareto-фронт NSGA-II — {fin_short_label} vs Качество интервалов\n"
        "Маршруты 20, 48, 55",
        fontsize=12, fontweight="bold",
    )
    plt.tight_layout()
    out_path = os.path.join(out_dir, "pareto_revenue_vs_mae.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Сохранён: {out_path}")
    created.append(out_path)

    # ── 1b. Копия графика с отображением всех рассмотренных вариантов ──────────
    if df_all is not None:
        fig, ax = plt.subplots(figsize=(10, 7))
        
        # 1. Сначала рисуем все рассмотренные варианты (полупрозрачные серые точки)
        ax.scatter(
            df_all[fin_col], df_all["headway_mae_min"],
            color="lightgray", alpha=0.6, s=40, edgecolors="gray", linewidths=0.3,
            label="Все рассмотренные варианты"
        )
        
        # 2. Поверх рисуем Парето-оптимальные решения (цветные точки)
        sc = ax.scatter(
            df[fin_col], df["headway_mae_min"],
            c=total_trams, cmap=cmap, norm=norm,
            s=80, edgecolors="k", linewidths=0.4, alpha=0.9,
            label="Парето-оптимальные решения"
        )
        
        _annotate_extremes(ax, fin_col, "headway_mae_min")
        
        ax.set_xlabel(f"{fin_label} →", fontsize=10)
        ax.set_ylabel("← MAE интервалов (мин) — меньше лучше", fontsize=10)
        ax.grid(True, alpha=0.3, linestyle="--")
        
        cbar = fig.colorbar(sc, ax=ax)
        cbar.set_label("Суммарный парк (трамваев) для Парето-решений", fontsize=9)
        
        ax.set_title(
            f"Все рассмотренные варианты vs Pareto-фронт\n"
            f"{fin_short_label} vs Качество интервалов (Маршруты 20, 48, 55)",
            fontsize=12, fontweight="bold",
        )
        ax.legend(loc="upper right", fontsize=9)
        plt.tight_layout()
        out_path_all = os.path.join(out_dir, "pareto_revenue_vs_mae_all.png")
        fig.savefig(out_path_all, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Сохранён график со всеми вариантами: {out_path_all}")
        created.append(out_path_all)

    # ── 2. Столбчатый график: распределение парка для Топ-10 решений ──────────
    top_n = min(10, len(df))
    # Сортируем решения по прибыли/выручке и берем первые top_n
    df_sorted = df.sort_values(fin_col, ascending=False).head(top_n).reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(12, 6))
    x_pos = np.arange(top_n)
    width = 0.25

    # Строим три столбца для каждого решения (по числу трамваев на каждом маршруте)
    bars_20 = ax.bar(x_pos - width, df_sorted["n_20"], width, label="Маршрут 20", color="#2196F3", alpha=0.85)
    bars_48 = ax.bar(x_pos,         df_sorted["n_48"], width, label="Маршрут 48", color="#4CAF50", alpha=0.85)
    bars_55 = ax.bar(x_pos + width, df_sorted["n_55"], width, label="Маршрут 55", color="#FF9800", alpha=0.85)

    # Добавляем текстовые подписи (Показатель / MAE) над каждой группой столбцов
    for i in range(top_n):
        total = df_sorted.loc[i, fin_col]
        mae   = df_sorted.loc[i, "headway_mae_min"]
        label_str = f"{total:.2f} млн\n{mae:.1f}м" if is_mp else f"{total/1000:.0f}к\n{mae:.1f}м"
        ax.text(i, df_sorted.loc[i, ["n_20", "n_48", "n_55"]].max() + 0.5,
                label_str,
                ha="center", fontsize=7, color="darkred")

    ax.set_xlabel(f"Решение (ранжировано по убыванию {fin_label_short_ru_genitive})", fontsize=11)
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

    # ── 3. Размер парка vs Прибыль/Доход с цветовой индикацией точности ─────────────
    if "total_trams" in df.columns:
        fig, ax = plt.subplots(figsize=(9, 6))
        # Цветовая шкала: RdYlGn_r (красный -> желтый -> зеленый, реверсивный). 
        # Зеленый — маленькая ошибка MAE (отлично), красный — большая ошибка.
        sc = ax.scatter(
            df["total_trams"], df[fin_col],
            c=df["headway_mae_min"], cmap="RdYlGn_r",
            s=80, edgecolors="k", linewidths=0.4, alpha=0.9,
        )
        ax.set_xlabel("Суммарный парк (трамваев) →", fontsize=10)
        ax.set_ylabel(f"{fin_label} →", fontsize=10)
        ax.set_title(
            f"Размер парка vs {fin_short_label} (цвет = MAE интервалов)",
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
        print("Использование: python plot_pareto.py <путь до pareto_front.csv> [out_dir] [n_max]")
        sys.exit(1)

    out_dir = sys.argv[2] if len(sys.argv) > 2 else None
    n_max = int(sys.argv[3]) if len(sys.argv) > 3 else None
    plot_pareto(csv_path=sys.argv[1], out_dir=out_dir, n_max=n_max)
