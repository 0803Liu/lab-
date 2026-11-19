import argparse
import csv
import hashlib
import json
import math
import os
import random
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import mlflow
import numpy as np
import optuna
import pandas as pd
import sklearn
import torch
import torch.nn as nn
from sklearn.datasets import load_digits
from torch.utils.data import DataLoader, TensorDataset
import yaml


def set_deterministic(seed: int, num_threads: int = 1):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(num_threads)
    torch.use_deterministic_algorithms(True)


def check_git_clean(allow_dirty: bool = False) -> str:
    status_output = subprocess.run(
        ["git", "status", "--porcelain"], capture_output=True, text=True
    ).stdout.strip()
    # 自动过滤掉 mlruns 目录，避免 MLflow 自动生成的文件干扰实验
    lines = [line for line in status_output.splitlines() if "mlruns" not in line]
    dirty = "\n".join(lines)
    if dirty and not allow_dirty:
        raise SystemExit("Есть незакоммиченные изменения — итоговый запуск невозможен:\n" + dirty)
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True
    ).stdout.strip()
    return commit or "no-commit"


def validate_config(cfg: dict):
    lr = cfg["train"]["lr"]
    if not (1e-6 <= lr <= 0.5):
        raise ValueError(f"Некорректный lr={lr}: допустимый диапазон [1e-6, 0.5].")
    if cfg["model"]["n_layers"] < 1 or cfg["model"]["hidden"] < 1:
        raise ValueError("n_layers и hidden должны быть положительными.")


class MLP(nn.Module):
    def __init__(self, n_layers: int, hidden: int, dropout: float):
        super().__init__()
        layers = []
        in_dim = 64
        for _ in range(n_layers):
            layers.append(nn.Linear(in_dim, hidden))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            in_dim = hidden
        layers.append(nn.Linear(in_dim, 10))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def load_data(split_path: str = "data/split.json"):
    p = Path(split_path)
    raw_bytes = p.read_bytes()
    sha256 = hashlib.sha256(raw_bytes).hexdigest()
    splits = json.loads(raw_bytes.decode("utf-8"))

    X, y = load_digits(return_X_y=True)
    X = (X / 16.0).astype(np.float32)
    y = y.astype(np.int64)

    return (
        (X[splits["train"]], y[splits["train"]], splits["train"]),
        (X[splits["val"]], y[splits["val"]], splits["val"]),
        (X[splits["test"]], y[splits["test"]], splits["test"]),
        sha256,
    )


def evaluate(model: nn.Module, X: np.ndarray, y: np.ndarray, criterion):
    model.eval()
    with torch.no_grad():
        logits = model(torch.from_numpy(X))
        loss = criterion(logits, torch.from_numpy(y)).item()
        preds = logits.argmax(dim=1).numpy()
    acc = float((preds == y).mean())
    return loss, acc, preds


def run_training(cfg: dict, eval_test: bool = False, allow_dirty: bool = False, nested: bool = False, trial=None):
    validate_config(cfg)
    commit = check_git_clean(allow_dirty=allow_dirty)

    seed = int(cfg["seed"])
    num_threads = int(cfg["runtime"].get("num_threads", 1))
    set_deterministic(seed, num_threads)

    (X_train, y_train, idx_train), (X_val, y_val, idx_val), (X_test, y_test, idx_test), split_sha = load_data()

    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    rand_hex = f"{random.getrandbits(48):012x}"
    run_id = f"{ts}_{rand_hex}"
    run_dir = Path("runs") / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    # 保存最终配置
    with open(run_dir / "config.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)

    start_time = datetime.now(timezone.utc).isoformat()
    meta = {
        "run_id": run_id,
        "series": cfg.get("series", "default"),
        "status": "running",
        "start_time": start_time,
        "end_time": None,
        "commit": commit,
        "python_version": sys.version.split()[0],
        "torch_version": torch.__version__,
        "sklearn_version": sklearn.__version__,
        "optuna_version": optuna.__version__,
        "mlflow_version": mlflow.__version__,
        "num_threads": num_threads,
        "seed": seed,
        "split_sha256": split_sha,
    }
    (run_dir / "metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment(cfg.get("experiment", "lr0"))

    history_path = run_dir / "history.csv"
    with open(history_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "val_loss", "val_acc", "epoch_time"])

    with mlflow.start_run(run_name=run_id, nested=nested):
        mlflow.set_tags({"commit": commit, "series": meta["series"], "seed": str(seed)})
        mlflow.log_params({
            "n_layers": cfg["model"]["n_layers"],
            "hidden": cfg["model"]["hidden"],
            "dropout": cfg["model"]["dropout"],
            "lr": cfg["train"]["lr"],
            "weight_decay": cfg["train"]["weight_decay"],
            "batch_size": cfg["train"]["batch_size"],
            "epochs": cfg["train"]["epochs"],
            "seed": seed,
        })

        try:
            g = torch.Generator()
            g.manual_seed(seed)
            train_ds = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
            train_loader = DataLoader(
                train_ds,
                batch_size=int(cfg["train"]["batch_size"]),
                shuffle=True,
                num_workers=0,
                generator=g,
            )

            model = MLP(
                n_layers=int(cfg["model"]["n_layers"]),
                hidden=int(cfg["model"]["hidden"]),
                dropout=float(cfg["model"]["dropout"]),
            )
            criterion = nn.CrossEntropyLoss()
            optimizer = torch.optim.Adam(
                model.parameters(),
                lr=float(cfg["train"]["lr"]),
                weight_decay=float(cfg["train"]["weight_decay"]),
            )

            epochs = int(cfg["train"]["epochs"])
            for epoch in range(1, epochs + 1):
                t0 = time.perf_counter()
                model.train()
                total_loss = 0.0
                for bx, by in train_loader:
                    optimizer.zero_grad()
                    out = model(bx)
                    loss = criterion(out, by)
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item() * len(by)
                train_loss = total_loss / len(X_train)

                val_loss, val_acc, val_preds = evaluate(model, X_val, y_val, criterion)
                epoch_time = time.perf_counter() - t0

                with open(history_path, "a", newline="", encoding="utf-8") as f:
                    csv.writer(f).writerow([epoch, train_loss, val_loss, val_acc, epoch_time])

                mlflow.log_metrics(
                    {"train_loss": train_loss, "val_loss": val_loss, "val_acc": val_acc},
                    step=epoch,
                )

                if trial is not None:
                    trial.report(val_acc, step=epoch)
                    if trial.should_prune():
                        meta["status"] = "pruned"
                        meta["end_time"] = datetime.now(timezone.utc).isoformat()
                        (run_dir / "metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
                        mlflow.set_tag("state", "PRUNED")
                        raise optuna.TrialPruned()

            # 保存 predictions.csv
            pred_rows = []
            for idx, yt, yp in zip(idx_val, y_val, val_preds):
                pred_rows.append(["val", int(idx), int(yt), int(yp)])

            if eval_test:
                test_loss, test_acc, test_preds = evaluate(model, X_test, y_test, criterion)
                for idx, yt, yp in zip(idx_test, y_test, test_preds):
                    pred_rows.append(["test", int(idx), int(yt), int(yp)])

            pred_path = run_dir / "predictions.csv"
            with open(pred_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["split", "index", "y_true", "y_pred"])
                writer.writerows(pred_rows)

            # 强制从 predictions.csv 与 history.csv 重新计算指标
            df_pred = pd.read_csv(pred_path)
            df_hist = pd.read_csv(history_path)
            val_sub = df_pred[df_pred["split"] == "val"]
            recalc_val_acc = float((val_sub["y_true"] == val_sub["y_pred"]).mean())
            total_train_time = float(df_hist["epoch_time"].sum())
            final_val_loss = float(df_hist["val_loss"].iloc[-1])

            metrics = {
                "val_acc": recalc_val_acc,
                "val_loss": final_val_loss,
                "train_time": total_train_time,
            }
            if eval_test:
                test_sub = df_pred[df_pred["split"] == "test"]
                metrics["test_acc"] = float((test_sub["y_true"] == test_sub["y_pred"]).mean())

            (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
            mlflow.log_metrics(metrics)
            mlflow.set_tag("state", "FAIL" if math.isnan(recalc_val_acc) else "COMPLETE")

            meta["status"] = "completed"
            meta["end_time"] = datetime.now(timezone.utc).isoformat()
            (run_dir / "metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

            for artifact in ["config.yaml", "metadata.json", "history.csv", "predictions.csv", "metrics.json"]:
                mlflow.log_artifact(str(run_dir / artifact))

            return run_id, metrics

        except optuna.TrialPruned:
            raise
        except Exception as e:
            meta["status"] = "failed"
            meta["end_time"] = datetime.now(timezone.utc).isoformat()
            meta["error"] = traceback.format_exc()
            (run_dir / "metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
            mlflow.set_tag("state", "FAIL")
            raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--series", type=str, default=None)
    parser.add_argument("--eval-test", action="store_true")
    parser.add_argument("--allow-dirty", action="store_true")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    if args.seed is not None:
        cfg["seed"] = args.seed
    if args.series is not None:
        cfg["series"] = args.series

    run_id, metrics = run_training(cfg, eval_test=args.eval_test, allow_dirty=args.allow_dirty)
    print(f"Run {run_id} finished: {metrics}")


if __name__ == "__main__":
    main()