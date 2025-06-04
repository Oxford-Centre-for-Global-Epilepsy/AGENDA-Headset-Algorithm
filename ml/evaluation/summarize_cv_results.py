import os
import glob
import pandas as pd
import numpy as np
import argparse

def summarize_folds(parent_dir, omit_list=None):
    all_dirs = glob.glob(os.path.join(parent_dir, "fold_*"))
    fold_dirs = []

    for d in all_dirs:
        basename = os.path.basename(d)
        if omit_list:
            tag = "omit_" + "_".join(omit_list.split(","))
            if basename.endswith(tag):
                fold_dirs.append(d)
        else:
            if "omit_" not in basename:
                fold_dirs.append(d)

    fold_dirs = sorted(fold_dirs)
    all_results = []

    print(fold_dirs)
    for fold_dir in fold_dirs:
        csv_path = os.path.join(fold_dir, "metrics_log.csv")
        if not os.path.exists(csv_path):
            print(f"⚠️ Skipping {fold_dir}: No metrics_log.csv found.")
            continue

        df = pd.read_csv(csv_path)
        best_row = df.loc[df['val_loss'].idxmin()].copy()
        best_row['fold'] = os.path.basename(fold_dir)
        all_results.append(best_row)

    if not all_results:
        raise ValueError(f"❌ No valid metrics found in the selected folds under {parent_dir}. Please check if 'metrics_log.csv' exists in the expected directories.")

    df_all = pd.DataFrame(all_results)
    metrics_only = df_all.drop(columns=["run_id", "fold", "epoch"])

    mean_metrics = metrics_only.mean()
    std_metrics = metrics_only.std()

    summary_df = pd.DataFrame({
        "metric": mean_metrics.index,
        "mean": mean_metrics.values,
        "std": std_metrics.values
    })

    omit_tag = "_".join(omit_list.split(",")) if omit_list else "all"
    summary_path = os.path.join(parent_dir, f"cv_summary_omit_{omit_tag}.csv")
    summary_df.to_csv(summary_path, index=False)

    print(f"✅ Saved summary to: {summary_path}")
    print("\n📊 Mean ± Std across folds:")
    print(summary_df.to_string(index=False, float_format="%.4f"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", required=True,
                        help="Path to directory containing fold_X folders")
    parser.add_argument("--omit_list", default="",
                        help="Comma-separated list of omitted electrodes, e.g. 'Fp1,Fp2'")
    args = parser.parse_args()

    summarize_folds(args.results_dir, omit_list=args.omit_list if args.omit_list else None)
