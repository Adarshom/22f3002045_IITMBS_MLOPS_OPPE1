import argparse, os, glob
import pandas as pd
import yaml

def load_params(p="params.yaml"):
    with open(p) as f:
        return yaml.safe_load(f)

def load_concat_csvs(folder):
    files = sorted(glob.glob(os.path.join(folder, "*.csv")))
    if not files:
        raise FileNotFoundError(f"No CSVs in {folder}")
    dfs = []
    for f in files:
        df = pd.read_csv(f)
        # Normalize columns if needed
        df.columns = [c.strip().lower() for c in df.columns]
        # expected: timestamp, open, high, low, close, volume, stock_name
        # allow 'symbol' as alias for stock_name
        if "stock_name" not in df.columns and "symbol" in df.columns:
            df["stock_name"] = df["symbol"]
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True)

def build_features(df, cfg):
    tcol = cfg["features"]["time_col"]
    ccol = cfg["features"]["close_col"]
    vcol = cfg["features"]["volume_col"]
    scol = cfg["features"]["stock_col"]
    w = int(cfg["features"]["roll_window"])
    h = int(cfg["features"]["horizon_min"])
    target_col = cfg["features"]["target_col"]

    # type & ordering
    df[tcol] = pd.to_datetime(df[tcol])
    # bring back canonical case for columns we’ll output
    df.rename(columns={
        tcol: "timestamp",
        ccol: "close",
        vcol: "volume",
        scol: "stock_name"
    }, inplace=True)
    df = df.sort_values(["stock_name", "timestamp"])

    # group-wise rolling; "last 10 available points" -> min_periods=1
    g = df.groupby("stock_name", group_keys=False)
    df["rolling_avg_10"] = g["close"].rolling(window=w, min_periods=1).mean().reset_index(level=0, drop=True)
    df["volume_sum_10"] = g["volume"].rolling(window=w, min_periods=1).sum().reset_index(level=0, drop=True)

    # future close at t+5 within each stock
    future_close = g["close"].shift(-h)
    df[target_col] = (future_close > df["close"]).astype("Int64")

    # drop tail rows with no future label
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
        feats_v01.to_parquet(args.out_v01, index=False)

if __name__ == "__main__":
    main()
