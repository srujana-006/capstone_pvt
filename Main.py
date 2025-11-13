from MAP_env import MAP
from Controller import Controller
import pandas as pd
import numpy as np
if __name__ == "__main__":
    # 1) Load grid environment
    env = MAP("Data/grid_config_2d.json")

    # 2) Initialise EVs (27 default)
    env.init_evs(seed=42)
    print(f"[Main] EVs initialised: {len(env.evs)} total.")

    # 3) Create controller
    ctrl = Controller(
        env=env,
        ticks_per_ep=180,  # 1 day of 8-min slots
        seed=123,
        csv_path="Data/5Years_SF_calls_latlong.csv",  # your dataset
        time_col="Received DtTm",
        lat_col="Latitude",
        lng_col="Longitude",
    )

    # 4) Run one episode to verify flow
    ctrl.run_one_episode()
