import matplotlib
matplotlib.use('Agg')  # 必须在 import pyplot 之前
import matplotlib.pyplot as plt

import json
from pathlib import Path
import numpy as np
import optuna
import pandas as pd
from collections import Counter


def load_runs():
    rows = []
    runs_dir = Path("runs")
    if runs_dir.exists():
        for run_dir in sorted(runs_dir.iterdir()):
            meta_f = run_dir / "metadata.json"
            met_f = run_dir / "metrics.json"
            if meta_f.exists() and met_f.exists():
                meta = json.loads(meta_f.read_text(encoding="utf-8"))
                met = json.loads(met_f.read_text(encoding="utf-8"))
                if meta["status"] == "completed":
                    rows.append({**meta, **met, "dir": str(run_dir)})
    return pd.DataFrame(rows)


def generate_param_importance_plot(study):
    """独立生成并保存超参数重要性条形图（图 2）——带强异常保护"""
    try:
        import optuna.importance
        comp_trials = [t for t in study.trials if t.state.name == 'COMPLETE']
        if len(comp_trials) < 2:
            print("[Notice] Not enough completed trials for parameter importance.")
            return

        imp = optuna.importance.get_param_importances(study)
        if not imp:
            return

        plt.figure(figsize=(7, 4), dpi=150)
        keys = list(imp.keys())[::-1]
        vals = [imp[k] for k in keys]
        plt.barh(keys, vals, color='tab:blue')
        plt.xlabel("Estimated relative importance")
        plt.title(f"Parameter importance ({len(comp_trials)} completed trials)")
        plt.grid(axis="x", alpha=0.3)
        plt.tight_layout()
        plt.savefig("reports/figures/fig2_param_importance.png")
        plt.close()
        print("图 2 已成功保存至 reports/figures/fig2_param_importance.png")
    except Exception as e:
        print(f"[Warning] Skipped param importance plot due to environment issue: {e}")


def main():
    # 创建报告所需的图表与源数据目录（对齐同学项目的工程规范）
    Path("reports/figures").mkdir(parents=True, exist_ok=True)
    source_data_dir = Path("reports/source_data")
    source_data_dir.mkdir(parents=True, exist_ok=True)

    df = load_runs()

    # 1. 实验 1：base_seeds (seed 1..5)
    base_s = df[df["series"] == "base_seeds"].sort_values("seed") if len(df) > 0 else pd.DataFrame()
    print("=== Эксперимент 1: base_seeds (1-5) ===")
    if len(base_s) > 0:
        print(base_s[["seed", "val_acc", "train_time", "run_id"]])
        accs = base_s["val_acc"].values * 100
        print(f"Min/Med/Max: {accs.min():.4f}% / {np.median(accs):.4f}% / {accs.max():.4f}%, Range: {accs.max()-accs.min():.4f} п.п.")

        # 导出实验 1 汇总数据到 source_data（对齐同学的项目规范）
        base_s[["seed", "val_acc", "train_time", "run_id"]].to_csv(source_data_dir / "base_seeds_summary.csv", index=False)

        # seed 1 vs seed 2 逐样本对比
        s1_row = base_s[base_s["seed"] == 1]
        s2_row = base_s[base_s["seed"] == 2]
        if len(s1_row) > 0 and len(s2_row) > 0:
            p1 = pd.read_csv(Path(s1_row["dir"].iloc[0]) / "predictions.csv")
            p2 = pd.read_csv(Path(s2_row["dir"].iloc[0]) / "predictions.csv")
            c1 = p1["y_true"] == p1["y_pred"]
            c2 = p2["y_true"] == p2["y_pred"]
            print(f"Seed 1 vs Seed 2: исправлено={(~c1 & c2).sum()}, испорчено={(c1 & ~c2).sum()}")
    else:
        print("[Notice] base_seeds runs not found.")

    # 2. 实验 2：Optuna 分析与绘图（使用原生遍历避开 trials_dataframe 的段错误）
    print("\n=== Эксперимент 2: Optuna ===")
    try:
        study = optuna.load_study(study_name="lr0_search", storage="sqlite:///optuna.db")
        states = [t.state.name for t in study.trials]
        print(Counter(states))

        exec_epochs = 0
        saved_time = 0.0
        completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
        pruned_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]

        for t in study.trials:
            steps = len(t.intermediate_values)
            if t.state == optuna.trial.TrialState.COMPLETE:
                steps = 30
            exec_epochs += steps
            if t.state == optuna.trial.TrialState.PRUNED and steps > 0 and t.datetime_complete and t.datetime_start:
                dur = (t.datetime_complete - t.datetime_start).total_seconds()
                saved_time += (30 - steps) * (dur / steps)
        print(f"Выполнено эпох: {exec_epochs} / 1200 ({100*(1 - exec_epochs/1200):.2f}% сэкономлено), ~{saved_time:.2f} с")

        # 图 1：Trial 准确率曲线
        plt.figure(figsize=(8, 4.5), dpi=150)
        comp_nums = [t.number for t in completed_trials]
        comp_vals = [t.value * 100 for t in completed_trials if t.value is not None]
        pruned_nums = [t.number for t in pruned_trials]
        pruned_vals = [t.value * 100 for t in pruned_trials if t.value is not None]

        plt.scatter(comp_nums, comp_vals, label="COMPLETE", zorder=3)
        plt.scatter(pruned_nums, pruned_vals, marker="x", label="PRUNED (intermediate)", zorder=2)
        if comp_vals:
            best_so_far = np.maximum.accumulate(comp_vals)
            plt.plot(comp_nums, best_so_far, color="green", label="Best completed trial so far")

        plt.xlabel("Trial number (zero-based)")
        plt.ylabel("Validation accuracy (%)")
        plt.title("LR0 hyperparameter search")
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig("reports/figures/fig1_optuna_trials.png")
        plt.close()
        print("图 1 已成功保存至 reports/figures/fig1_optuna_trials.png")

        # 调用图 2 生成函数（超参数重要性 fANOVA 图）
        generate_param_importance_plot(study)

    except Exception as e:
        print(f"[Warning] Optuna analysis error: {e}")

    # 3. 实验 3：final_base vs final_best (seed 101..105)
    print("\n=== Эксперимент 3: Сравнение final_base и final_best ===")
    if len(df) > 0:
        fb = df[df["series"] == "final_base"].sort_values("seed")
        fbest = df[df["series"] == "final_best"].sort_values("seed")
        if len(fb) > 0 and len(fbest) > 0:
            fixed_total, broken_total = 0, 0
            paired_rows = []
            for s in [101, 102, 103, 104, 105]:
                match_b = fb[fb["seed"] == s]
                match_best = fbest[fbest["seed"] == s]
                if len(match_b) > 0 and len(match_best) > 0:
                    rb = match_b.iloc[0]
                    rbest = match_best.iloc[0]
                    d_val = (rbest["val_acc"] - rb["val_acc"]) * 100
                    d_test = (rbest["test_acc"] - rb["test_acc"]) * 100
                    print(f"Seed {s}: Base test={rb['test_acc']*100:.4f}%, Best test={rbest['test_acc']*100:.4f}%, d_val={d_val:+.4f}, d_test={d_test:+.4f}")

                    paired_rows.append({
                        "seed": s,
                        "base_test_acc": rb["test_acc"],
                        "best_test_acc": rbest["test_acc"],
                        "d_val_pp": d_val,
                        "d_test_pp": d_test
                    })

                    pb = pd.read_csv(Path(rb["dir"]) / "predictions.csv").query("split == 'test'")
                    pbest = pd.read_csv(Path(rbest["dir"]) / "predictions.csv").query("split == 'test'")
                    cb = pb["y_true"] == pb["y_pred"]
                    cbest = pbest["y_true"] == pbest["y_pred"]
                    fixed_total += (~cb & cbest).sum()
                    broken_total += (cb & ~cbest).sum()
            
            print(f"Суммарно по test: исправлено={fixed_total}, испорчено={broken_total}, чистый выигрыш={fixed_total - broken_total}")

            # 导出实验 3 配对比较汇总到 source_data（对齐同学的项目规范）
            pd.DataFrame(paired_rows).to_csv(source_data_dir / "paired_comparison.csv", index=False)
            print("实验 3 配对汇总表已成功导出至 reports/source_data/paired_comparison.csv")

        else:
            print("[Notice] final_base or final_best runs not found yet. Please run Experiment 3 training first.")
    else:
        print("[Notice] No completed runs found.")


if __name__ == "__main__":
    main()