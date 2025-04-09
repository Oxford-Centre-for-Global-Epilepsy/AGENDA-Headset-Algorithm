import os
import shutil
import torch
import pandas as pd

def export_best_models(root_dir="outputs", export_dir="exports/best_models"):
    os.makedirs(export_dir, exist_ok=True)
    summary = []

    for name in os.listdir(root_dir):
        fold_path = os.path.join(root_dir, name)
        if not os.path.isdir(fold_path):
            continue

        best_model_path = os.path.join(fold_path, "best_model.pt")
        if not os.path.exists(best_model_path):
            continue

        try:
            ckpt = torch.load(best_model_path, map_location="cpu")
            epoch = ckpt.get("epoch", None)
            val_loss = ckpt.get("best_val_loss", None)
        except Exception as e:
            print(f"⚠️ Failed to load {best_model_path}: {e}")
            continue

        # Extract fold number from folder name (e.g., run_fold_3 → 3)
        try:
            fold_index = int("".join(filter(str.isdigit, name)))
        except ValueError:
            fold_index = name

        export_name = f"fold_{fold_index}_best_model.pt"
        export_path = os.path.join(export_dir, export_name)
        shutil.copy2(best_model_path, export_path)

        summary.append({
            "fold": fold_index,
            "val_loss": val_loss,
            "epoch": epoch,
            "model_path": export_path
        })

    df = pd.DataFrame(summary).sort_values("fold")
    summary_csv = os.path.join(export_dir, "fold_summary.csv")
    df.to_csv(summary_csv, index=False)

    print(f"✅ Exported {len(df)} models to {export_dir}")
    print(f"📄 Summary saved to {summary_csv}")
    return df

if __name__ == "__main__":
    export_best_models()
