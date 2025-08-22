from ml_tflm.dataset.eeg_dataset_rewrite import EEGRecordingDatasetTF

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import welch

def plot_eeg_segment(segment, channel_names=None, offset=5, figsize=(10, 8), title="EEG Segment"):
    """
    Plots a single EEG segment as stacked traces.

    Args:
        segment (np.ndarray): EEG segment of shape (channels, timepoints), e.g., (21, 128).
        channel_names (list of str, optional): List of channel names. Defaults to "Ch 1", "Ch 2", etc.
        offset (float): Vertical spacing between channels.
        figsize (tuple): Size of the figure.
        title (str): Title of the plot.
    """
    num_channels, num_timepoints = segment.shape
    time = np.arange(num_timepoints)

    if channel_names is None:
        channel_names = [f"Ch {i+1}" for i in range(num_channels)]

    plt.figure(figsize=figsize)
    
    for i in range(num_channels):
        plt.plot(time, segment[i] + i * offset, label=channel_names[i])

    plt.title(title)
    plt.xlabel("Time (samples)")
    plt.ylabel("Amplitude + offset")
    plt.yticks([])
    plt.legend(loc="upper right", fontsize="small", ncol=2)
    plt.tight_layout()
    plt.show()

def plot_eeg_psd(segment, fs=128, channel_names=None, nperseg=128, figsize=(10, 6)):
    """
    Plots the Power Spectral Density (PSD) for each channel in a single EEG segment.

    Args:
        segment (np.ndarray): EEG data of shape (channels, timepoints), e.g., (21, 128).
        fs (int): Sampling frequency in Hz.
        channel_names (list of str): Optional list of channel names.
        nperseg (int): Length of each FFT segment.
        figsize (tuple): Size of the figure.
    """
    num_channels = segment.shape[0]
    if channel_names is None:
        channel_names = [f"Ch {i+1}" for i in range(num_channels)]

    plt.figure(figsize=figsize)

    for i in range(num_channels):
        f, Pxx = welch(segment[i], fs=fs, nperseg=nperseg)
        plt.semilogy(f, Pxx, label=channel_names[i])

    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Power Spectral Density (V²/Hz)")
    plt.title("PSD of EEG Segment (per channel)")
    plt.legend(loc="upper right", fontsize="small", ncol=2)
    plt.grid(True, which="both", ls="--", alpha=0.4)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    label_config = {
        "label_map": {"neurotypical": 0, "epileptic": 1, "focal": 2, "generalized": 3, "left": 4, "right": 5},
        "inverse_label_map": {0: "neurotypical", 1: "epileptic", 2: "focal", 3: "generalized", 4: "left", 5: "right"},
    }

    dataset = EEGRecordingDatasetTF(
        h5_file_path="ml_tflm/dataset/agenda_data_01/combined_south_africa_monopolar_standard_10_20.h5",
        dataset_name="combined_south_africa_monopolar_standard_10_20",
        label_config=label_config, 
        mirror_flag=False
        )

    sample = dataset[100]['data']

    #plot_eeg_segment(sample[10])

    plot_eeg_psd(sample[10])

    dataset.close()
