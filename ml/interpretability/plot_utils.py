import matplotlib.pyplot as plt
import numpy as np

def plot_gradcam_heatmap(heatmap, 
                         input_eeg=None, 
                         channel_names=None, 
                         title="Grad-CAM Heatmap",
                         cmap="viridis",
                         save_path=None):
    """
    Plots a Grad-CAM heatmap over time (or channels & time if 2D).

    Args:
        heatmap (np.ndarray): Grad-CAM output; shape [T'] or [F, T'] or [C, T']
        input_eeg (np.ndarray, optional): Original EEG data [C, T] for overlay
        channel_names (list of str, optional): EEG channel names
        title (str): Plot title
        cmap (str): Colormap to use
        save_path (str, optional): If provided, saves the plot to this path
    """
    plt.figure(figsize=(10, 4))

    if heatmap.ndim == 1:
        plt.plot(heatmap, label="Grad-CAM", color="r")
        if input_eeg is not None:
            plt.plot(np.mean(input_eeg, axis=0), alpha=0.5, label="Mean EEG")
        plt.xlabel("Time")
        plt.ylabel("Activation")
        plt.legend()
    elif heatmap.ndim == 2:
        plt.imshow(heatmap, aspect="auto", cmap=cmap, origin="lower")
        plt.colorbar(label="Activation")
        plt.xlabel("Time")
        plt.ylabel("Channels" if channel_names is None else "EEG Channel")
        if channel_names is not None:
            plt.yticks(ticks=np.arange(len(channel_names)), labels=channel_names)

    plt.title(title)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"Saved Grad-CAM heatmap to {save_path}")
    else:
        plt.show()

def plot_saliency_map(saliency_map,
                      title="Saliency Map",
                      xlabel="Time",
                      ylabel="EEG Channels",
                      channel_names=None,
                      cmap="plasma",
                      save_path=None):
    """
    Plots 1D or 2D saliency map.

    Args:
        saliency_map (np.ndarray): [T] or [C, T]
        title (str): Plot title
        xlabel (str): x-axis label
        ylabel (str): y-axis label
        channel_names (list): Optional EEG channel labels
        cmap (str): Colormap to use
        save_path (str): Optional path to save plot
    """
    plt.figure(figsize=(10, 4))

    if saliency_map.ndim == 1:
        plt.plot(saliency_map, color="darkorange")
        plt.ylabel("Saliency")
    else:
        plt.imshow(saliency_map, aspect="auto", cmap=cmap, origin="lower")
        plt.colorbar(label="Saliency")
        if channel_names is not None:
            plt.yticks(ticks=np.arange(len(channel_names)), labels=channel_names)

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"Saved saliency plot to {save_path}")
    else:
        plt.show()