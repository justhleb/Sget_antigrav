"""
Точка входа CLI для мультимаршрутной симуляции.

Примеры:
    python -m simulation.runner --routes 20 48 55
    python -m simulation.runner --routes 20 48 55 --trams 30 30 30
    python -m simulation.runner --routes 20 --no-plots
    python -m simulation.runner --routes 20 48 --configs-dir my_configs
"""
import argparse
import logging
import sys
from pathlib import Path

from simulation.multi_route import MultiRouteSimulation

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)

DEFAULT_CONFIGS_DIR  = Path("configs")
DEFAULT_TRAMS_PER_ROUTE = 30


def resolve_route_pairs(
    route_ids: list[str], configs_dir: Path
) -> dict[str, tuple[str, str]]:
    pairs   = {}
    missing = []

    for route_id in route_ids:
        fwd = configs_dir / f"route_{route_id}_fwd_config.json"
        bwd = configs_dir / f"route_{route_id}_bwd_config.json"

        if not fwd.exists():
            missing.append(str(fwd))
        if not bwd.exists():
            missing.append(str(bwd))

        if fwd.exists() and bwd.exists():
            pairs[route_id] = (str(fwd), str(bwd))

    if missing:
        log.error("Не найдены конфиги:")
        for m in missing:
            log.error(f"  ✗ {m}")
        sys.exit(1)

    log.info(f"Найдено маршрутов: {len(pairs)}")
    for route_id, (fwd, bwd) in pairs.items():
        log.info(f"  ✓ маршрут {route_id}: {Path(fwd).name} + {Path(bwd).name}")

    return pairs


def main():
    parser = argparse.ArgumentParser(
        description="Мультимаршрутная симуляция трамваев",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog=(
            "Примеры использования:\n"
            "  python -m simulation.runner --routes 20 48 55\n"
            "  python -m simulation.runner --routes 20 48 55 --trams 30 30 30\n"
            "  python -m simulation.runner --routes 20 --no-plots\n"
            "  python -m simulation.runner --routes 20 48 --configs-dir my_configs\n"
        )
    )
    parser.add_argument(
        "--routes", nargs="+", required=True, metavar="ROUTE_ID",
        help="Номера маршрутов (например: 20 48 55)"
    )
    parser.add_argument(
        "--trams", nargs="+", type=int, metavar="N",
        help=(
            "Кол-во трамваев на каждый маршрут в том же порядке что --routes.\n"
            "Можно указать одно число — применится ко всем маршрутам.\n"
            f"По умолчанию: {DEFAULT_TRAMS_PER_ROUTE} на каждый маршрут.\n"
            "Примеры: --trams 30  или  --trams 25 30 35"
        )
    )
    parser.add_argument(
        "--configs-dir", type=Path, default=DEFAULT_CONFIGS_DIR, metavar="DIR",
        help=f"Папка с конфиг-файлами (по умолчанию: {DEFAULT_CONFIGS_DIR})"
    )
    parser.add_argument("--no-plots", action="store_true", help="Не создавать графики")
    parser.add_argument("--no-logs",  action="store_true", help="Не сохранять CSV-логи")
    args = parser.parse_args()

    if not args.configs_dir.exists():
        log.error(f"Папка конфигов не найдена: {args.configs_dir}")
        sys.exit(1)

    route_pairs = resolve_route_pairs(args.routes, args.configs_dir)
    n_routes    = len(route_pairs)

    # Разбираем --trams
    if args.trams is None:
        tram_counts = [DEFAULT_TRAMS_PER_ROUTE] * n_routes
    elif len(args.trams) == 1:
        tram_counts = args.trams * n_routes          # одно число → на все маршруты
    elif len(args.trams) == n_routes:
        tram_counts = args.trams
    else:
        log.error(
            f"--trams должен содержать 1 или {n_routes} значений, "
            f"получено: {len(args.trams)}"
        )
        sys.exit(1)

    log.info("Распределение трамваев:")
    for route_id, count in zip(route_pairs.keys(), tram_counts):
        log.info(f"  маршрут {route_id}: {count} трамваев")

    sim = MultiRouteSimulation(route_pairs, tram_counts=tram_counts)
    sim.run(
        plot_graphs=not args.no_plots,
        save_logs=not args.no_logs,
    )


if __name__ == "__main__":
    main()
