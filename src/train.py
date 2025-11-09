import argparse, json, os, joblib
import pandas as pd
import mlflow, yaml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score

def load_params(p="params.yaml"):
    with open(p) as f: 
        return yaml.safe_load(f)

def make_features(df: pd.DataFrame, target_col: str):
    """Build X, y from engineered parquet."""
    # base numeric features
    X = df[["rolling_avg_10", "volume_sum_10", "close", "volume"]].copy()
    # stock_name one-hot (lightweight)
    X = pd.concat([X, pd.get_dummies(df["stock_name"], prefix="sn", dtype=float)], axis=1)
    y = df[target_col].astype(int)
    # Drop any rows with NaNs after engineering
    mask = ~X.isna().any(axis=1)
    X, y = X.loc[mask], y.loc[mask]
    return X, y

def train_one(dataset_path, model_out, metrics_out, cfg, tag, predictions_out=None):
    df = pd.read_parquet(dataset_path)
    target = cfg["features"]["target_col"]
    X, y = make_features(df, target)

    # Stratified split by label (simple & adequate for the exam)
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=cfg["train"]["test_size"], random_state=cfg["train"]["random_state"], stratify=y
    )

    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", cfg["mlflow"]["uri"]))
    mlflow.set_experiment(os.getenv("MLFLOW_EXPERIMENT", cfg["mlflow"]["experiment"]))

    # Standardize numerics before LR; handle imbalance
    pipe = make_pipeline(
        StandardScaler(with_mean=False),  # works with sparse one-hot
        LogisticRegression(
            C=float(cfg["train"]["lr"]["C"]),
            max_iter=int(cfg["train"]["lr"]["max_iter"]),
            class_weight="balanced",
            n_jobs=-1,
        )
    )

    with mlflow.start_run(tags={"iteration": tag}):
        mlflow.log_params({
            "C": float(cfg["train"]["lr"]["C"]),
            "max_iter": int(cfg["train"]["lr"]["max_iter"]),
            "class_weight": "balanced",
            "scaler": "StandardScaler(with_mean=False)"
        })

        pipe.fit(Xtr, ytr)
        proba = pipe.predict_proba(Xte)[:, 1]
        preds = (proba >= 0.5).astype(int)

        acc = float(accuracy_score(yte, preds))
        try:
            auc = float(roc_auc_score(yte, proba))
        except ValueError:
            auc = None

        mlflow.log_metric("accuracy", acc)
        if auc is not None:
            mlflow.log_metric("auc", auc)

        os.makedirs(os.path.dirname(model_out), exist_ok=True)
        joblib.dump(pipe, model_out)
        mlflow.log_artifact(model_out)

        # optional predictions dump (for CI/CML)
        if predictions_out:
            pd.DataFrame({"prediction": preds, "proba": proba}).to_csv(predictions_out, index=False)

        with open(metrics_out, "w") as f:
            json.dump({"accuracy": acc, "auc": auc, "n_train": int(len(ytr)), "n_test": int(len(yte))}, f, indent=2)

def main():
    cfg = load_params()
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--model_out", required=True)
    ap.add_argument("--metrics_out", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--predictions_out", default=os.getenv("PREDICTIONS_OUT"))
    args = ap.parse_args()
    train_one(args.dataset, args.model_out, args.metrics_out, cfg, args.tag, args.predictions_out)

if __name__ == "__main__":
    main()
