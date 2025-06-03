
import torch
import torch.nn as nn
import torch.optim as optim
from darts.models.darts_model import DARTSModel
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from ml.datasets.utils import get_cv_fold
import os
import yaml

def extract_final_architecture(model):
    architecture = []
    for layer_idx, layer in enumerate(model.cells):
        op_names = [op.__class__.__name__ for op in layer.ops]
        weights = layer.alpha.detach().cpu().numpy()
        best_op_idx = weights.argmax()
        best_op = op_names[best_op_idx]
        architecture.append({"layer": layer_idx, "operation": best_op})
    return architecture

def main():
    config = OmegaConf.load("darts/config/search_config.yaml")
    torch.manual_seed(config.search.seed)

    train_set, val_set = get_cv_fold(config.search.dataset, config.search.fold_index)
    train_loader = DataLoader(train_set, batch_size=config.search.batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=config.search.batch_size)

    model = DARTSModel(C=config.search.num_channels, num_classes=2, num_layers=config.search.num_layers,
                       op_names=config.search.operation_choices)

    optimizer = optim.Adam(model.parameters(), lr=config.search.learning_rate)
    arch_optimizer = optim.Adam(model.arch_parameters(), lr=config.search.arch_learning_rate)
    criterion = nn.CrossEntropyLoss()

    os.makedirs(config.search.log_dir, exist_ok=True)

    for epoch in range(config.search.epochs):
        model.train()
        for x, y in train_loader:
            optimizer.zero_grad()
            arch_optimizer.zero_grad()

            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()

            optimizer.step()
            arch_optimizer.step()

        print(f"[Epoch {epoch}] Train Loss: {loss.item():.4f}")

    # After training loop
    final_arch = extract_final_architecture(model)
    with open("outputs/darts/final_architecture.yaml", "w") as f:
        yaml.dump(final_arch, f)

    # Save final architecture weights
    torch.save(model.state_dict(), os.path.join(config.search.log_dir, "final_model.pt"))

if __name__ == "__main__":
    main()
