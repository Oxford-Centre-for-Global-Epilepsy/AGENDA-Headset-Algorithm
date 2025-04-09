import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE
from umap import UMAP

def project_features(features, labels=None, method="umap", n_components=2, random_state=42):
    """
    Reduces features to 2D using UMAP or t-SNE.

    Args:
        features (np.ndarray): [N, D] array of pooled feature vectors
        labels (list or np.ndarray): [N] class labels (optional, for coloring)
        method (str): 'umap' or 'tsne'
        n_components (int): Output dimensionality
        random_state (int): Random seed

    Returns:
        projected (np.ndarray): [N, 2] reduced 2D features
    """
    if isinstance(features, torch.Tensor):
        features = features.detach().cpu().numpy()

    reducer = UMAP(n_components=n_components, random_state=random_state) if method == "umap" \
        else TSNE(n_components=n_components, random_state=random_state)

    return reducer.fit_transform(features)


def plot_projection(projected_features, labels=None, label_names=None, title="Feature Projection", save_path=None):
    """
    Plots 2D projected features with optional class coloring.

    Args:
        projected_features (np.ndarray): [N, 2] array from UMAP/t-SNE
        labels (list or np.ndarray): Class labels for coloring
        label_names (dict): Optional mapping from label index to name
        title (str): Plot title
        save_path (str): If given, saves to file
    """
    plt.figure(figsize=(6, 6))
    if labels is not None:
        labels = np.array(labels)
        unique_labels = np.unique(labels)
        for lbl in unique_labels:
            idxs = np.where(labels == lbl)[0]
            label_name = label_names[lbl] if label_names and lbl in label_names else str(lbl)
            plt.scatter(projected_features[idxs, 0], projected_features[idxs, 1], label=label_name, alpha=0.7)
        plt.legend()
    else:
        plt.scatter(projected_features[:, 0], projected_features[:, 1], alpha=0.6)

    plt.title(title)
    plt.xlabel("Dim 1")
    plt.ylabel("Dim 2")
    plt.grid(True)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"Saved projection plot to {save_path}")
    else:
        plt.show()
