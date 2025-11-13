# build_grid_config.py
import argparse
from datetime import datetime
from utils.Helpers import load_calls, compute_edges_2d, save_grid_config_2d



def main():
    ap = argparse.ArgumentParser(description="Build 2D grid configuration (lat/lng edges).")
    ap.add_argument("--csv", required=True, help="CSV file containing Latitude and Longitude columns.")
    ap.add_argument("--lat", default="Latitude", help="Column name for latitude.")
    ap.add_argument("--lng", default="Longitude", help="Column name for longitude.")
    ap.add_argument("--speed", type=float, default=40.0, help="Average travel speed (km/h).")
    ap.add_argument("--slot", type=float, default=8.0, help="Timeslot length in minutes.")
    ap.add_argument("--out", required=True, help="Output JSON file path.")
    args = ap.parse_args()

    df = load_calls(args.csv)
    if args.lat not in df.columns or args.lng not in df.columns:
        raise ValueError(f"CSV must contain columns '{args.lat}' and '{args.lng}'.")

    lat_edges, lng_edges = compute_edges_2d(
        df,
        avg_speed_kmph=args.speed,
        slot_mins=args.slot,
        lat_col=args.lat,
        lng_col=args.lng,
    )

    meta = {
        "source_csv": args.csv,
        "lat_col": args.lat,
        "lng_col": args.lng,
        "avg_speed_kmph": args.speed,
        "slot_minutes": args.slot,
        "created_at": datetime.utcnow().isoformat() + "Z",
    }

    save_grid_config_2d(args.out, lat_edges, lng_edges, meta)

    print(f"Wrote {args.out}")
    print(f"Lat range: [{lat_edges[0]:.5f}, {lat_edges[-1]:.5f}]")
    print(f"Lng range: [{lng_edges[0]:.5f}, {lng_edges[-1]:.5f}]")
    print(f"Rows: {len(lat_edges) - 1}, Cols: {len(lng_edges) - 1}")


if __name__ == "__main__":
    main()
