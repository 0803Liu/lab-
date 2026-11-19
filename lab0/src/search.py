import argparse
import math
from pathlib import Path
import mlflow
import optuna
from optuna.study import MaxTrialsCallback
import yaml
from train import check_git_clean, run_training


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-trials", type=int, default=40, help="目标总 trial 数，例如先设 10 测试中断再设 40")
    args = parser.parse_args()

    check_git_clean(allow_dirty=False)
    with open("configs/base.yaml", "r", encoding="utf-8") as f:
        base_cfg = yaml.safe_load(f)

    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("lr0")

    def objective(trial: optuna.Trial):
        cfg = yaml.safe_load(yaml.safe_dump(base_cfg))
        cfg["series"] = "optuna_search"
        cfg["seed"] = 0
        cfg["train"]["lr"] = trial.suggest_float("lr", 1e-4, 1e-1, log=True)
        cfg["model"]["n_layers"] = trial.suggest_int("n_layers", 1, 3)
        cfg["model"]["hidden"] = trial.suggest_int("hidden", 32, 512, log=True)
        cfg["model"]["dropout"] = trial.suggest_float("dropout", 0.0, 0.5)
        cfg["train"]["weight_decay"] = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)
        cfg["train"]["batch_size"] = trial.suggest_categorical("batch_size", [32, 64, 128])

        _, metrics = run_training(cfg, eval_test=False, allow_dirty=False, nested=True, trial=trial)
        return metrics["val_acc"]

    study = optuna.create_study(
        study_name="lr0_search",
        storage="sqlite:///optuna.db",
        load_if_exists=True,
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=5),
    )

    with mlflow.start_run(run_name="lr0_search"):
        study.optimize(
            objective,
            n_trials=args.n_trials,
            callbacks=[MaxTrialsCallback(args.n_trials, states=None)],
        )

    # 导出最佳配置到 configs/best.yaml
    best = study.best_trial
    best_cfg = yaml.safe_load(yaml.safe_dump(base_cfg))
    best_cfg["series"] = "best"
    best_cfg["model"]["n_layers"] = int(best.params["n_layers"])
    best_cfg["model"]["hidden"] = int(best.params["hidden"])
    best_cfg["model"]["dropout"] = float(best.params["dropout"])
    best_cfg["train"]["lr"] = float(best.params["lr"])
    best_cfg["train"]["weight_decay"] = float(best.params["weight_decay"])
    best_cfg["train"]["batch_size"] = int(best.params["batch_size"])

    with open("configs/best.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(best_cfg, f, sort_keys=False)
    print(f"Best trial #{best.number}: val_acc={best.value:.6f}, saved to configs/best.yaml")


if __name__ == "__main__":
    main()