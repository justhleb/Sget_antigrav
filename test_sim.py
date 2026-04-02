import logging
from simulation.multi_route import MultiRouteSimulation

ROUTE_PAIRS = {
    "20": ("configs/route_20_fwd_config.json", "configs/route_20_bwd_config.json"),
    "48": ("configs/route_48_fwd_config.json", "configs/route_48_bwd_config.json"),
    "55": ("configs/route_55_fwd_config.json", "configs/route_55_bwd_config.json"),
}

sim = MultiRouteSimulation(ROUTE_PAIRS, tram_counts=[10, 10, 10], run_dir="outputs/run_test")
sim.run(plot_graphs=True, save_logs=False)
