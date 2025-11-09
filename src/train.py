import argparse, json, os, joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
import mlflow, yaml

def load_params(p="params.yaml"):
    with open(p) as f:
        return yaml.safe_load(f)

def train_one(dataset_path, model_out, metrics_out, cfg, tag):
    df = pd.read_parquet(dataset_path)
    target = cfg["features"]["target_col"]

    # Features per spec + raw signals
    X = df[["rolling_avg_10", "volume_sum_10", "close", "volume"]].copy()
    # stock_name one-hot
    X = pd.concat([X, pd.get_dummies(df["stock_name"], prefix="sn", dtype=float)], axis=1)
    y = df[target].astype(int)

    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=cfg["train"]["test_size"], random_state=cfg["train"]["random_state"], stratify=y
    )

    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", cfg["mlflow"]["uri"]))
    mlflow.set_experiment(os.getenv("MLFLOW_EXPERIMENT", cfg["mlflow"]["experiment"]))
    with mlflow.start_run(tags={"iteration": tag}):
        C = float(cfg["train"]["lr"]["C"]); max_iter = int(cfg["train"]["lr"]["max_iter"])
        mlflow.log_params({"C": C, "max_iter": max_iter, "iteration": tag})

        model = LogisticRegression(C=C, max_iter=max_iter, n_jobs=-1)
        model.fit(Xtr, ytr)
        proba = model.predict_proba(Xte)[:, 1]
        preds = (proba >= 0.5).astype(int)

        acc = float(accuracy_score(yte, preds))
        try: auc = float(roc_auc_score(yte, proba))
        except ValueError: auc = None

        mlflow.log_metric("accuracy", acc)
        if auc is not None: mlflow.log_metric("auc", auc)

        os.makedirs(os.path.dirname(model_out), exist_ok=True)
        joblib.dump(model, model_out)
        mlflow.log_artifact(model_out)

        with open(metrics_out, "w") as f:
            json.dump({"accuracy": acc, "auc": auc, "n_train": int(len(ytr)), "n_test": int(len(yte))}, f, indent=2)

def main():
    cfg = load_params()
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--model_out", required=True)
    ap.add_argument("--metrics_out", required=True)
    ap.add_argument("--tag", required=True)
    args = ap.parse_args()
    train_one(args.dataset, args.model_out, args.metrics_out, cfg, args.tag)

if __name__ == "__main__":
    main()
