from omegaconf import OmegaConf
import os
import sys
import ast
import torch
import random
import numpy as np
from torch.utils.data import DataLoader
import mlflow
import mlflow.pytorch

from ml.datasets.eeg_dataset import EEGRecordingDataset
from ml.training.losses import HierarchicalLoss
from ml.models.hierarchical_classifier import EEGNetHierarchicalClassifier


def parse_config():
    if "--config" not in sys.argv:
        raise ValueError("Please provide a config file using --config path/to/file.yaml")

    config_index = sys.argv.index("--config")
    config_path = sys.argv[config_index + 1]
    overrides = sys.argv[config_index + 2:]

    base_config = OmegaConf.load(config_path)
    cli_config = OmegaConf.from_dotlist(overrides)
    config = OmegaConf.merge(base_config, cli_config)

    if not config.get("run_id"):
        config.run_id = "final_model"

    base_data_dir = os.environ.get("DATA")
    if base_data_dir is None:
        raise EnvironmentError("❌ The DATA environment variable is not set on this system.")

    config.output_dir = os.path.join(
        base_data_dir,
        "AGENDA-Headset-Algorithm/outputs",
        config.experiment_name,
        config.run_id
    )

    omit = config.dataset.get("drop_electrodes")
    if isinstance(omit, str):
        try:
            config.dataset.drop_electrodes = ast.literal_eval(omit)
        except (ValueError, SyntaxError):
            config.dataset.drop_electrodes = [ch.strip() for ch in omit.split(',')]

    return config


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    print(f"🔒 Set global random seed to: {seed}")


def main():
    config = parse_config()
    os.makedirs(config.output_dir, exist_ok=True)

    config.device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"✅ Using device: {config.device}")

    set_seed(config.seed)

    mlflow.set_tracking_uri("file://" + os.path.abspath("mlruns"))
    mlflow.set_experiment(config.experiment_name)

    with mlflow.start_run(run_name="final_model"):
        mlflow.log_params({
            "experiment_name": config.experiment_name,
            "dataset_name": config.dataset.dataset_name,
            "dropped_electrodes": config.dataset.drop_electrodes,
            "batch_size": config.train.batch_size,
            "epochs": config.train.epochs,
            "learning_rate": config.train.learning_rate,
            "dropout_rate": config.model.dropout_rate,
            "kernel_length": config.model.kernel_length,
            "pooling_type": config.model.pooling_type
        })

        label_map = {
            'neurotypical': 0,
            'epileptic': 1,
            'focal': 2,
            'generalized': 3,
            'left': 4,
            'right': 5
        }

        split_path = os.path.join(config.dataset.subject_split_dir, "test_split.json")
        test_splits = set(np.load(split_path, allow_pickle=True).item()["test"])

        # Load full training set (excluding test subjects)
        all_subjects = set(os.listdir(config.dataset.dataset_path))
        train_subjects = sorted(list(all_subjects - test_splits))

        train_dataset = EEGRecordingDataset(
            config.dataset.dataset_path, config.dataset.dataset_name, label_map,
            omit_channels=config.dataset.drop_electrodes,
            subject_ids=train_subjects
        )

        train_loader = DataLoader(train_dataset, batch_size=config.train.batch_size, shuffle=True)

        num_channels = train_dataset.get_num_channels()
        print(f"Using {num_channels} EEG Channels")

        eegnet_args = {
            "num_channels": num_channels,
            "num_samples": 129,
            "dropout_rate": config.model.dropout_rate,
            "kernel_length": config.model.kernel_length
        }

        pooling_args = {"hidden_dim": 64} if config.model.pooling_type == "attention" else {}
        model = EEGNetHierarchicalClassifier(
            eegnet_args=eegnet_args,
            pooling_type=config.model.pooling_type,
            pooling_args=pooling_args
        ).to(config.device)

        criterion = HierarchicalLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=config.train.learning_rate)

        model.train()
        for epoch in range(config.train.epochs):
            total_loss = 0
            for batch in train_loader:
                x = batch["data"].to(config.device)
                y = batch["labels"].to(config.device)
                m = batch["label_mask"].to(config.device)
                attn_mask = batch["attention_mask"].to(config.device)

                out = model(x, attention_mask=attn_mask)
                loss = criterion(out, y, m)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            avg_loss = total_loss / len(train_loader)
            print(f"Epoch {epoch+1}/{config.train.epochs} - Train Loss: {avg_loss:.4f}")
            mlflow.log_metric("train_loss", avg_loss, step=epoch + 1)

        mlflow.pytorch.log_model(model, "model")
        torch.save(model.state_dict(), os.path.join(config.output_dir, "final_model.pt"))
        print("✅ Final model trained and saved!")


if __name__ == "__main__":
    main()