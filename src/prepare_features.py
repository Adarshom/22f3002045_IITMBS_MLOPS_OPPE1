import argparse, os, glob, re
import pandas as pd
import yaml

def load_params(p="params.yaml"):
    with open(p) as f:
        return yaml.safe_load(f)

def _infer_stock_name_from_path(path: str) -> str:
    base = os.path.basename(path)
    # take token before first "__" or first dot as fallback
    m = re.match(r"([^_]+)__", base)
    return (m.group(1) if m else os.path.splitext(base)[0]).upper()

def _read_and_standardize(path: str) -> pd.DataFrame:
    """
    Expect columns: timestamp, open, high, low, close, volume (as in your samples).
    Add stock_name from filename. Parse tz-aware timestamps, convert to UTC (naive).
    """
    df = pd.read_csv(path)
    # normalize headers
    df.columns = [c.strip().lower() for c in df.columns]
    required = {"timestamp", "close", "volume"}
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f"{os.path.basename(path)} missing columns: {missing}")

    # timestamp: parse with timezone, convert to UTC, then drop tz to keep naive (stable for joins)
    ts = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
    df["timestamp"] = ts.dt.tz_convert("UTC").dt.tz_localize(None)

    # numeric coercion
    for col in ("open", "high", "low", "close", "volume"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # add stock_name
    df["stock_name"] = _infer_stock_name_from_path(path)

    # drop duplicates & obvious bad rows
    df = df.dropna(subset=["timestamp", "close"]).drop_duplicates(subset=["stock_name", "timestamp"])

    return df[["timestamp", "stock_name", "close", "volume"]].copy()

def load_concat_csvs(folder: str) -> pd.DataFrame:
    files = sorted(glob.glob(os.path.join(folder, "*.csv")))
    if not files:
        raise FileNotFoundError(f"No CSVs in {folder}")
    parts = []
    for f in files:
        parts.append(_read_and_standardize(f))
    return pd.concat(parts, ignore_index=True)

def build_features(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """
    Features:
      - rolling_avg_10 over close (last 10 available points)
      - volume_sum_10 over volume (last 10 available points)
      - keep raw close, volume, stock_name
    Target:
      - will_up_in_5m := 1 if close(t+5) > close(t), else 0
    """
    w = int(cfg["features"]["roll_window"])
    h = int(cfg["features"]["horizon_min"])
    target_col = cfg["features"]["target_col"]

    df = df.sort_values(["stock_name", "timestamp"]).copy()

    g = df.groupby("stock_name", group_keys=False)
    df["rolling_avg_10"] = g["close"].rolling(window=w, min_periods=1).mean().reset_index(level=0, drop=True)
    df["volume_sum_10"] = g["volume"].rolling(window=w, min_periods=1).sum().reset_index(level=0, drop=True)

    # t+5 future comparison within each stock
    future_close = g["close"].shift(-h)
    df[target_col] = (future_close > df["close"]).astype("Int64")

    # drop rows where we cannot form a label (tail)
    df = df.dropna(subset=[target_col])

    keep = ["timestamp", "stock_name", "close", "volume", "rolling_avg_10", "volume_sum_10", target_col]
    return df[keep]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--v0", required=True)
    ap.add_argument("--v1", required=False, default=None)
    ap.add_argument("--out_v0", required=True)
    ap.add_argument("--out_v01", required=True)
    ap.add_argument("--params", default="params.yaml")
    args = ap.parse_args()

    cfg = load_params(args.params)

    v0 = load_concat_csvs(args.v0)
    feats_v0 = build_features(v0, cfg)
    os.makedirs(os.path.dirname(args.out_v0), exist_ok=True)
    feats_v0.to_parquet(args.out_v0, index=False)

    if args.v1 and os.path.isdir(args.v1):
        v1 = load_concat_csvs(args.v1)
        both = pd.concat([v0, v1], ignore_index=True)
        feats_v01 = build_features(both, cfg)
        os.makedirs(os.path.dirname(args.out_v01), exist_ok=True)
        feats_v01.to_parquet(args.out_v01, index=False)

if __name__ == "__main__":
    main()
