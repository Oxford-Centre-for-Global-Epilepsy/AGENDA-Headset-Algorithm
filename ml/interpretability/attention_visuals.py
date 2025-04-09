import matplotlib.pyplot as plt
import numpy as np

def plot_attention_weights(attn_weights, 
                           attention_mask=None,
                           title="Attention Weights Over Epochs",
                           ylabel="Attention Weight",
                           xlabel="Epoch",
                           save_path=None,
                           predicted_label=None,
                           true_label=None):
    """
    Plots attention weights over epochs for a single recording.

    Args:
        attn_weights (Tensor or np.ndarray): Shape [E]; attention weights for each epoch.
        attention_mask (Tensor or np.ndarray, optional): Mask of valid epochs (bool or 0/1).
        title (str): Plot title.
        ylabel (str): Label for y-axis.
        xlabel (str): Label for x-axis.
        save_path (str): If given, saves plot to this path.
        predicted_label (str, optional): Optional predicted label to show in title.
        true_label (str, optional): Optional ground truth label to show in title.
    """
    if isinstance(attn_weights, torch.Tensor):
        attn_weights = attn_weights.detach().cpu().numpy()
    if attention_mask is not None and isinstance(attention_mask, torch.Tensor):
        attention_mask = attention_mask.detach().cpu().numpy()

    epochs = np.arange(len(attn_weights))

    plt.figure(figsize=(12, 3))
    plt.plot(epochs, attn_weights, color="darkred", lw=2, label="Attention Weight")

    if attention_mask is not None:
        valid_epochs = np.where(attention_mask == 1)[0]
        plt.fill_between(epochs, 0, 1, where=attention_mask.astype(bool), alpha=0.1, color="green", label="Valid Epochs")

    if predicted_label or true_label:
        title += f"\nPredicted: {predicted_label}, True: {true_label}"

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.ylim([0, 1.05])
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"Saved attention weight plot to {save_path}")
    else:
        plt.show()