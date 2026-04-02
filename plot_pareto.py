# plot_pareto.py
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sys
import os


def plot_pareto(csv_path: str, out_dir: str = None):
    df = pd.read_csv(csv_path)
    out_dir = out_dir or os.path.dirname(csv_path) or "."

    # ── Фикс знаков: pymoo минимизирует всё, поэтому km и passengers сохранены
    #    с инвертированным знаком — возвращаем в человекочитаемый вид
    if "total_tram_km" in df.columns:
        df["total_tram_km"] = df["total_tram_km"].abs()
    if "total_served" in df.columns:
        df["total_served"] = df["total_served"].abs()

    has_passengers = "total_served" in df.columns

    total_trams = df[["n_20", "n_48", "n_55"]].sum(axis=1).values
    norm  = plt.Normalize(total_trams.min(), total_trams.max())
    cmap  = "viridis"

    def _scatter(ax, x_col, y_col, x_label, y_label):
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
        candidates = {
            "макс km":    df[x_col].idxmax() if "km" in x_col   else None,
            "мин mae":    df["headway_mae_min"].idxmin()        if "mae" in x_col or "mae" in y_col else None,
            "макс пасс":  df[y_col].idxmax()  if "served" in y_col else None,
        }
        for label, idx in candidates.items():
            if idx is None:
                continue
            ax.annotate(
                f'{label}\n[{int(df.loc[idx,"n_20"])},{int(df.loc[idx,"n_48"])},{int(df.loc[idx,"n_55"])}]',
                xy=(df.loc[idx, x_col], df.loc[idx, y_col]),
                xytext=(10, 10), textcoords="offset points",
                fontsize=7, color="darkred",
                arrowprops=dict(arrowstyle="->", color="darkred", lw=0.8),
            )

    # ── 3 попарных 2D графика ─────────────────────────────────────────────────
    pairs = [
        ("total_tram_km",   "headway_mae_min",
         "Трамвай-км (транспортная работа) →",
         "← MAE интервалов (мин) — меньше лучше",
         "km_vs_mae"),

        ("total_tram_km",   "total_served",
         "Трамвай-км (транспортная работа) →",
         "Обслужено пассажиров →",
         "km_vs_passengers"),

        ("total_served",    "headway_mae_min",
         "Обслужено пассажиров →",
         "← MAE интервалов (мин) — меньше лучше",
         "passengers_vs_mae"),
    ]

    # убираем пары где нет колонки
    if not has_passengers:
        pairs = [p for p in pairs if "served" not in p[0] and "served" not in p[1]]

    created = []
    for x_col, y_col, x_label, y_label, suffix in pairs:
        fig, ax = plt.subplots(figsize=(9, 6))
        sc = _scatter(ax, x_col, y_col, x_label, y_label)
        _annotate_extremes(ax, x_col, y_col)

        cbar = fig.colorbar(sc, ax=ax)
        cbar.set_label("Суммарный парк (трамваев)", fontsize=9)

        ax.set_title(
            f"Pareto-фронт NSGA-II — маршруты 20, 48, 55\n",
            fontsize=11,
        )
        plt.tight_layout()

        out_path = os.path.join(out_dir, f"pareto_{suffix}.png")
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Сохранён: {out_path}")
        created.append(out_path)

    # ── 3D график если есть все три метрики ───────────────────────────────────
    if has_passengers:
        fig = plt.figure(figsize=(12, 8))
        ax3d = fig.add_subplot(111, projection="3d")

        sc = ax3d.scatter(
            df["total_tram_km"], df["headway_mae_min"], df["total_served"],
            c=total_trams, cmap=cmap, norm=norm,
            s=80, edgecolors="k", linewidths=0.4, alpha=0.9,
        )

        # аннотации крайних точек
        extremes = {
            "макс km":   df["total_tram_km"].idxmax(),
            "мин mae":   df["headway_mae_min"].idxmin(),
            "макс пасс": df["total_served"].idxmax(),
        }
        for label, idx in extremes.items():
            ax3d.text(
                df.loc[idx, "total_tram_km"],
                df.loc[idx, "headway_mae_min"],
                df.loc[idx, "total_served"],
                f'  {label}\n  [{int(df.loc[idx,"n_20"])},{int(df.loc[idx,"n_48"])},{int(df.loc[idx,"n_55"])}]',
                fontsize=7, color="darkred",
            )

        cbar = fig.colorbar(sc, ax=ax3d, pad=0.1, shrink=0.6)
        cbar.set_label("Суммарный парк (трамваев)", fontsize=9)

        ax3d.set_xlabel("Трамвай-км →", fontsize=9, labelpad=10)
        ax3d.set_ylabel("MAE интервалов (мин) →", fontsize=9, labelpad=10)
        ax3d.set_zlabel("Пассажиры →", fontsize=9, labelpad=10)
        ax3d.set_title(
            "Pareto-фронт NSGA-II — маршруты 20, 48, 55\n",
            fontsize=11, pad=15,
        )
        ax3d.view_init(elev=20, azim=45)
        plt.tight_layout()

        out_path = os.path.join(out_dir, "pareto_front_3d.png")
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Сохранён: {out_path}")
        created.append(out_path)

    print(f"\nВсего графиков: {len(created)}")
    return created


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Использование: python plot_pareto.py <путь до pareto_front.csv> [out_dir]")
        sys.exit(1)

    out_dir = sys.argv[2] if len(sys.argv) > 2 else None
    plot_pareto(csv_path=sys.argv[1], out_dir=out_dir)
