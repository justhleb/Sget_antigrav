import logging
logging.getLogger("simulation").setLevel(logging.ERROR)

from simulation.multi_route import MultiRouteSimulation

ROUTE_PAIRS = {
    "20": ("configs/route_20_fwd_config.json", "configs/route_20_bwd_config.json"),
    "48": ("configs/route_48_fwd_config.json", "configs/route_48_bwd_config.json"),
    "55": ("configs/route_55_fwd_config.json", "configs/route_55_bwd_config.json"),
}

print("=== Тест с минимальным парком [5, 5, 5] ===")
sim = MultiRouteSimulation.from_params(ROUTE_PAIRS, tram_counts=[5, 5, 5])
sim.run(plot_graphs=False, save_logs=False)
_, km, mae, _ = sim.get_objectives()
print(f"total_km={km:.1f},  headway_mae={mae:.2f} мин")

print()
print("=== Тест с нормальным парком [15, 12, 14] ===")
sim2 = MultiRouteSimulation.from_params(ROUTE_PAIRS, tram_counts=[15, 12, 14])
sim2.run(plot_graphs=False, save_logs=False)
_, km2, mae2, _ = sim2.get_objectives()
print(f"total_km={km2:.1f},  headway_mae={mae2:.2f} мин")
