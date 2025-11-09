import os, json, yaml, pandas as pd, mlflow
from itertools import product
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

def load_cfg():
    with open("params.yaml") as f:
        return yaml.safe_load(f)

def load_xy(path, target_col):
    df = pd.read_parquet(path)
    X = df[["rolling_avg_10","volume_sum_10","close","volume"]].copy()
    X = pd.concat([X, pd.get_dummies(df["stock_name"], prefix="sn", dtype=float)], axis=1)
    y = df[target_col].astype(int)
    mask = ~X.isna().any(axis=1)
    return X.loc[mask], y.loc[mask]

def main():
    cfg = load_cfg()
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", cfg["mlflow"]["uri"]))
    mlflow.set_experiment(os.getenv("MLFLOW_EXPERIMENT", cfg["mlflow"]["experiment"]))
    model_name = os.getenv("REGISTRY_MODEL_NAME", "stock_5m_lr")

    X, y = load_xy(cfg["data"]["feats_v01"], cfg["features"]["target_col"])
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=cfg["train"]["test_size"],
        random_state=cfg["train"]["random_state"], stratify=y
    )

    grid = {"C":[0.1,1.0,3.0], "max_iter":[200,400]}
    best = (-1.0, None)   # (acc, run_id)

    for C, max_iter in product(grid["C"], grid["max_iter"]):
        with mlflow.start_run(tags={"iteration":"v01","hpo":"grid"}):
            mlflow.log_params({"C":float(C), "max_iter":int(max_iter), "class_weight":"balanced"})
            m = LogisticRegression(C=float(C), max_iter=int(max_iter), class_weight="balanced", n_jobs=-1)
            m.fit(Xtr, ytr)
            acc = accuracy_score(yte, (m.predict_proba(Xte)[:,1] >= 0.5).astype(int))
            mlflow.log_metric("accuracy", float(acc))
            mlflow.sklearn.log_model(m, artifact_path="model")
            rid = mlflow.active_run().info.run_id
            if acc > best[0]:
                best = (acc, rid)

    # Register best
    res = mlflow.register_model(f"runs:/{best[1]}/model", model_name)
    print(json.dumps({"best_acc": best[0], "best_run_id": best[1], "registered_model": model_name, "version": res.version}))
if __name__ == "__main__":
    main()
