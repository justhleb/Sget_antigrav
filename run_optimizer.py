from optimizer import run_nsga2, N_MAX, ROUTE_PAIRS
import os
from datetime import datetime

ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
out_dir = os.path.join("outputs", f"run_{ts}", "nsga2")

run_nsga2(
    n_max=N_MAX,
    pop_size=20,
    n_gen=8,
    seed=42,
    out_dir=out_dir,
)
