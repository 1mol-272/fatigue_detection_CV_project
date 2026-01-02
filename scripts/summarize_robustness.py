import os
import re
import glob
import argparse
import pandas as pd

EXP_RE = re.compile(r"ws(?P<ws>\d+)_be(?P<be>\d+(?:\.\d+)?)_ym(?P<ym>\d+(?:\.\d+)?)_mf(?P<mf>\d+)")

def parse_exp_name(name: str):
    m = EXP_RE.match(name)
    if not m:
        return {}
    d = m.groupdict()
    return {
        "window_size": int(d["ws"]),
        "blink_ear_thresh": float(d["be"]),
        "yawn_mar_thresh": float(d["ym"]),
        "min_frames": int(d["mf"]),
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_root", default="results/experiments", help="Folder containing <exp>/metrics_test_all.csv")
    ap.add_argument("--out_all_csv", default="results/summary/all_models_all_exps.csv")
    ap.add_argument("--out_summary_csv", default="results/summary/model_robustness_summary.csv")
    ap.add_argument("--metric", default="f1_macro", choices=["f1_macro","accuracy","precision_macro","recall_macro"])
    args = ap.parse_args()

    exp_dirs = [d for d in glob.glob(os.path.join(args.results_root, "*")) if os.path.isdir(d)]
    rows = []

    for exp_dir in exp_dirs:
        exp_name = os.path.basename(exp_dir)
        p = os.path.join(exp_dir, "metrics_test_all.csv")
        if not os.path.exists(p):
            continue

        df = pd.read_csv(p)
        meta = parse_exp_name(exp_name)
        df["exp"] = exp_name
        for k, v in meta.items():
            df[k] = v
        rows.append(df)

    if not rows:
        raise RuntimeError(f"No metrics_test_all.csv found under: {args.results_root}")

    all_df = pd.concat(rows, ignore_index=True)
    os.makedirs(os.path.dirname(args.out_all_csv), exist_ok=True)
    all_df.to_csv(args.out_all_csv, index=False)
    print("Saved:", args.out_all_csv)

    # robustness summary
    g = all_df.groupby("model")[args.metric]
    summary = pd.DataFrame({
        "mean": g.mean(),
        "std": g.std(ddof=0),
        "median": g.median(),
        "min": g.min(),
        "max": g.max(),
        "n_exps": g.count(),
    }).reset_index()

    # win_rate: for each exp, mark model(s) with best metric
    best_per_exp = all_df.loc[all_df.groupby("exp")[args.metric].idxmax(), ["exp", "model"]]
    win_counts = best_per_exp["model"].value_counts()
    summary["wins"] = summary["model"].map(win_counts).fillna(0).astype(int)
    summary["win_rate"] = summary["wins"] / summary["n_exps"]

    summary = summary.sort_values(["mean", "win_rate"], ascending=False)
    os.makedirs(os.path.dirname(args.out_summary_csv), exist_ok=True)
    summary.to_csv(args.out_summary_csv, index=False)
    print("Saved:", args.out_summary_csv)
    print(summary.head(10))

if __name__ == "__main__":
    main()
