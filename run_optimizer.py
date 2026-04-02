from optimizer import run_nsga2
import os
from datetime import datetime

ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
out_dir = os.path.join("outputs", f"run_{ts}", "nsga2")

run_nsga2(
    n_max=30,
    pop_size=20,
    n_gen=8,
    seed=42,
    out_dir=out_dir,
)
