import pandas as pd, yaml
from src.prepare_features import build_features

def test_rolling_and_target_creation():
    cfg = yaml.safe_load(open("params.yaml"))
    # 20 minutes of synthetic data for a single stock
    df = pd.DataFrame({
        "timestamp": pd.date_range("2021-01-01 09:15", periods=20, freq="T"),
        "close": list(range(20)),
        "volume": [1]*20,
        "stock_name": ["S1"]*20
    })
    out = build_features(df.copy(), cfg)
    assert "rolling_avg_10" in out.columns
    assert "volume_sum_10" in out.columns
    # min_periods logic: first avg equals first close
    assert float(out["rolling_avg_10"].iloc[0]) == float(out["close"].iloc[0])
    # target should exist and have only 0/1
    assert out[cfg["features"]["target_col"]].dropna().isin([0,1]).all()
