# utils/Helpers.py
import json
import math
import numpy as np
import pandas as pd
from typing import Tuple,List,Dict, Any
import re

# -------------------------------------------------------------
# Normalization bounds (global constants)
# -------------------------------------------------------------
W_MIN, W_MAX = 0.0, 48.99
E_MIN, E_MAX = 0.0, 25.2126
P_MIN, P_MAX = 0.0, 30.98
H_MIN, H_MAX = 0.74, 34.87

# -------------------------------------------------------------
# BASIC LOAD/SAVE
# -------------------------------------------------------------
def load_calls(csv_path: str) -> pd.DataFrame:
    return pd.read_csv(csv_path)


def save_grid_config_2d(path: str, lat_edges: np.ndarray, lng_edges: np.ndarray, metadata: Dict[str, Any]) -> None:
    data = {
        "schema": "grid_config_2d_v1",
        "metadata": metadata,
        "lat_edges": lat_edges.tolist(),
        "lng_edges": lng_edges.tolist(),
        "n_rows": len(lat_edges) - 1,
        "n_cols": len(lng_edges) - 1,
    }
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def load_grid_config_2d(path: str) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    with open(path) as f:
        cfg = json.load(f)
    return np.array(cfg["lat_edges"]), np.array(cfg["lng_edges"]), cfg.get("metadata", {})


# -------------------------------------------------------------
# GRID EDGE COMPUTATION
# -------------------------------------------------------------
def compute_edges_2d(
    df: pd.DataFrame,
    avg_speed_kmph: float = 40,
    slot_mins: float = 8,
    lat_col: str = "Latitude",
    lng_col: str = "Longitude",
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute grid edges from min/max latitude and longitude."""
    min_lat, max_lat = df[lat_col].min(), df[lat_col].max()
    min_lng, max_lng = df[lng_col].min(), df[lng_col].max()

    # Convert km to degrees
    km_radius = (avg_speed_kmph * slot_mins) / 60.0
    deg_per_km_lat = 1 / 111.0
    deg_per_km_lng = 1 / (111.0 * math.cos(math.radians((min_lat + max_lat) / 2.0)))

    lat_step = km_radius * deg_per_km_lat
    lng_step = km_radius * deg_per_km_lng

    lat_edges = np.arange(min_lat, max_lat + lat_step, lat_step)
    lng_edges = np.arange(min_lng, max_lng + lng_step, lng_step)

    return lat_edges, lng_edges


# -------------------------------------------------------------
# GRID INDEX MAPPING
# -------------------------------------------------------------
def point_to_grid_index(lat: float, lng: float, lat_edges: np.ndarray, lng_edges: np.ndarray) -> int:
    """Return 1D grid index for a (lat, lng) point."""
    row = np.searchsorted(lat_edges, lat, side="right") - 1
    col = np.searchsorted(lng_edges, lng, side="right") - 1

    n_rows = len(lat_edges) - 1
    n_cols = len(lng_edges) - 1

    # Clamp to valid range
    row = max(0, min(row, n_rows - 1))
    col = max(0, min(col, n_cols - 1))

    return int(row * n_cols + col)

# utils/Helpers.py (append these)



# WKT "POINT (lng lat)" single-row parser
_wkt_point_re = re.compile(r"POINT\s*\(\s*([-+]?\d+(\.\d+)?)\s+([-+]?\d+(\.\d+)?)\s*\)", re.IGNORECASE)
def parse_wkt_row(val: str) -> Tuple[float, float] | None:
    if not isinstance(val, str):
        return None
    m = _wkt_point_re.search(val)
    if not m:
        return None
    lng = float(m.group(1)); lat = float(m.group(3))
    return (lat, lng)  # return (lat, lng) order

def to_tick_index(ts: pd.Timestamp) -> int:
    # 24h / 8min = 180 slots
    mins = ts.hour * 60 + ts.minute
    idx = mins // 8
    return int(max(0, min(179, idx)))

def build_daily_incident_schedule(
    df: pd.DataFrame,
    day: pd.Timestamp,
    time_col: str = "Received DtTm",
    lat_col: str | None = "Latitude",
    lng_col: str | None = "Longitude",
    wkt_col: str | None = None,
) -> Dict[int, List[Tuple[float, float]]]:
    """
    Returns {tick: [(lat,lng), ...], ...} for a single calendar day in df.
    If lat/lng not present, set wkt_col to parse 'POINT (lng lat)'.
    """
    tmp = df.copy()
    tmp[time_col] = pd.to_datetime(tmp[time_col], errors="coerce")
    day_start = pd.Timestamp(day.normalize())
    day_end = day_start + pd.Timedelta(days=1)
    tmp = tmp[(tmp[time_col] >= day_start) & (tmp[time_col] < day_end)]

    coords: List[Tuple[pd.Timestamp, float, float]] = []

    if lat_col and lng_col and lat_col in tmp.columns and lng_col in tmp.columns:
        tmp = tmp.dropna(subset=[lat_col, lng_col])
        for ts, lat, lng in zip(tmp[time_col], tmp[lat_col], tmp[lng_col]):
            if pd.notna(lat) and pd.notna(lng):
                coords.append((ts, float(lat), float(lng)))
    elif wkt_col and wkt_col in tmp.columns:
        for ts, w in zip(tmp[time_col], tmp[wkt_col]):
            p = parse_wkt_row(w)
            if p:
                lat, lng = p
                coords.append((ts, lat, lng))
    else:
        return {}

    schedule: Dict[int, List[Tuple[float, float]]] = {i: [] for i in range(180)}
    for ts, lat, lng in coords:
        t = to_tick_index(ts)
        schedule[t].append((lat, lng))
    return schedule

# -----------------------------
# Reward / Utility helpers
# -----------------------------
#Function safe norm is not required
'''
def _safe_norm(x: float, xmin: float, xmax: float, invert: bool = False) -> float:
    """Map x in [xmin, xmax] → [0,1]. Clamp outside. If invert=True, flip to 1 - norm."""
    if xmax <= xmin:
        return 0.0
    x = max(min(x, xmax), xmin)
    n = (x - xmin) / (xmax - xmin)
    return 1.0 - n if invert else n
'''
#Navigation utility
def utility_navigation(W_busy: float, H_min: float = 0.0, H_max: float = H_MAX) -> float:
  if W_busy < H_min:
    U_N =1
  elif H_min<W_busy<H_max:
    U_N = (W_busy-H_min)/(H_max-H_min)
  else:
    U_N =0
  return U_N

#Dispatch utilities
def utility_dispatch_v(W_idle: float, W_min: float = 0.0, W_max: float = W_MAX) -> float:
  #U_V calcluation
  if W_idle>W_max:
    U_V = W_idle/W_max
  else:
    U_V =1
  return U_V
def utility_dispatch_p(W_kt: float, P_min: float = 0.0, P_max: float = P_MAX) -> float:
  #U_P calcluation
  if W_kt>P_max:
    U_P = W_kt/P_max
  else:
    U_P =1
  return U_P
def utility_dispatch_total(W_idle: float, W_kt: float, beta: float = 0.5, W_min: float = 0.0,
    W_max: float = 1.0, P_min: float = 0.0, P_max: float = 1.0) -> float:
  U_V = utility_dispatch_v(W_idle)
  U_P = utility_dispatch_p(W_kt)
  U_D = beta * U_V + (1-beta)*U_P
  return U_D

#Repositioning utility
def utility_repositioning(W_idle: float, E_idle: float, alpha: float = 0.5, W_min: float = 0.0, W_max: float = W_MAX,
    E_min: float = 0.0, E_max: float = E_MAX) -> float:
    #U_RW calcluation
    if W_idle < W_min:
        U_RW =1
    elif W_min<W_idle<W_max:
        U_RW = (W_idle-W_min)/(W_max-W_min) # To be verified with U_RW = (W_max-W_idle)/(W_max-W_min)
    else:
        U_RW =0
    #U_RE calculation
    if E_idle < E_min:
        U_RE =1
    elif E_min<E_idle<E_max:
        U_RE = (E_max-E_idle)/(E_max-E_min)
    else:
        U_RE = 0
    #overall U_R
    alpha=0.5
    U_R = alpha * U_RW + (1.0-alpha)*U_RE
    return U_R

'''
#Testing
W_BUSY = 20.54, W_KT = 8.25, W_IDLE = 27.14, E_IDLE = 15.20
alpha = 0.5, beta=0.5 
print("Dispatch utility V: ", utility_dispatch_v(W_IDLE, W_MIN, W_MAX))
print("Dispatch utility P: ", utility_dispatch_p(W_KT, P_MIN,P_MAX))
print("Dispatch utility Total: ", utility_dispatch_total(W_IDLE, W_KT, beta, W_MIN, W_MAX, P_MIN, P_MAX))
print("Navigation utility: ", utility_navigation(W_BUSY, H_MIN, H_MAX))
print("Repositioning utility: ", utility_repositioning(W_IDLE,E_IDLE,alpha,W_MIN,W_MAX,E_MIN,E_MAX))
'''

# ---- Optional: simple reward wrappers you can call during learning ----

def reward_dispatch(W_idle: float, W_kt: float) -> float:
    """Reward for dispatching based on combined vehicle + patient utility."""
    from utils.Helpers import W_MIN, W_MAX, P_MIN, P_MAX
    U_D = utility_dispatch_total(W_idle, W_kt, W_min=W_MIN, W_max=W_MAX, P_min=P_MIN, P_max=P_MAX)
    return U_D 


def reward_navigation(W_busy: float) -> float:
    """
    Reward for navigation decisions. High congestion (busy) => lower reward.
    Scale to [-1, +1]: R = 2*U_N - 1
    """
    U_N = utility_navigation(W_busy)
    return U_N 


def reward_reposition(
    W_idle: float,
    E_idle: float,
    move_cost_norm: float = 0.0,
    cost_weight: float = 0.2,
) -> float:
    """
    Reward for repositioning:
      base = 2*U_R - 1  in [-1, +1]
      penalty = cost_weight * move_cost_norm   in [0, cost_weight]
      R = base - penalty
    """
    U_R = utility_repositioning(W_idle, E_idle)
    base = 2.0 * U_R - 1.0
    penalty = max(0.0, min(1.0, move_cost_norm)) * max(0.0, cost_weight)
    return base - penalty

import re
_WKT_POINT_RE = re.compile(r"POINT\s*\(\s*([\-0-9\.]+)\s+([\-0-9\.]+)\s*\)", re.IGNORECASE)

def wkt_point_to_lat_lng(wkt: str) -> tuple[float, float]:
    """
    Parse 'POINT (x y)' -> (lat, lng).
    WKT uses x = longitude, y = latitude.
    """
    if not isinstance(wkt, str):
        raise ValueError("WKT must be a string like 'POINT (x y)'")
    m = _WKT_POINT_RE.search(wkt)
    if not m:
        raise ValueError(f"Bad WKT POINT: {wkt!r}")
    x, y = float(m.group(1)), float(m.group(2))
    return (y, x)  # (lat, lng)

def add_lat_lng_from_point(df, point_col: str = "point",
                           lat_col: str = "Latitude", lng_col: str = "Longitude"):
    """
    Return a copy of df with Latitude/Longitude added from a WKT POINT column.
    """
    import pandas as pd
    lats, lngs = [], []
    for v in df[point_col].fillna(""):
        lat, lng = wkt_point_to_lat_lng(v)
        lats.append(lat); lngs.append(lng)
    out = df.copy()
    out[lat_col] = lats
    out[lng_col] = lngs
    return out

# ---------- NAVIGATION HELPERS ----------

import math

def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat/2)**2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon/2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c

def travel_minutes(lat1: float, lon1: float, lat2: float, lon2: float, kmph: float = 40.0) -> float:
    """ETA in minutes at constant average speed."""
    km = haversine_km(lat1, lon1, lat2, lon2)
    return 60.0 * km / max(kmph, 1e-6)
