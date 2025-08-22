import mne
import numpy as np
import re

import os
from pathlib import Path

import random

import matplotlib.pyplot as plt

from mne.time_frequency import psd_array_welch

def load_edf_as_numpy(file_path):
    """
    Load EEG data from an EDF file and return as a NumPy array.
    
    Parameters:
        file_path (str): Path to the .edf EEG file.
        
    Returns:
        data (np.ndarray): EEG data array of shape (n_channels, n_samples).
        times (np.ndarray): Time values in seconds (1D array).
        ch_names (list): List of channel names.
        sfreq (float): Sampling frequency in Hz.
    """
    raw = mne.io.read_raw_edf(file_path, preload=True)
    data, times = raw.get_data(return_times=True)
    return data, times, raw.info['ch_names'], raw.info['sfreq']

def extract_21_channels(data, ch_names):
    """
    Extract and return the EEG data and names for standard 10–20 electrodes.
    Also handles alias conversion (T3->T7, etc).
    """
    standard_10_20 = {
        "fp1", "fp2", 
        "f7", "f3", "fz", "f4", "f8",
        "a1", "t7", "c3", "cz", "c4", "t8", "a2",
        "p7", "p3", "pz", "p4", "p8",
        "o1", "o2"
    }

    alias_map = {
        "t3": "t7", "t4": "t8",
        "t5": "p7", "t6": "p8"
    }

    matched_indices = []
    matched_names = []

    for i, orig in enumerate(ch_names):
        name = orig.lower().strip()
        name = re.sub(r'^eeg[\s\-]*', '', name)
        name = re.sub(r'[^a-z0-9]', '', name)

        # Alias substitution
        name = alias_map.get(name, name)

        if name in standard_10_20:
            matched_indices.append(i)
            matched_names.append(orig)

    if not matched_indices:
        print("[WARN] No matching 10–20 channels found.")
        return np.zeros((0, data.shape[1])), []

    return data[matched_indices, :], matched_names

def plot_eeg_segment(data, ch_names, sfreq, start_time=0.0):
    """
    Plot EEG data segment.

    Args:
        data (np.ndarray): EEG data, shape (channels, timepoints)
        ch_names (List[str]): Names of the channels
        sfreq (float): Sampling frequency
        start_time (float): Starting time in seconds (for x-axis labeling)
    """
    num_ch, num_pts = data.shape
    time = np.linspace(start_time, start_time + num_pts / sfreq, num_pts)

    fig, ax = plt.subplots(figsize=(12, 0.4 * num_ch))
    offset = 5 * np.std(data)  # Vertical offset between traces

    for i in range(num_ch):
        ax.plot(time, data[i] + i * offset, label=ch_names[i])

    ax.set_yticks([i * offset for i in range(num_ch)])
    ax.set_yticklabels(ch_names)
    ax.set_xlabel("Time (s)")
    ax.set_title("EEG Segment")
    plt.tight_layout()
    plt.show()

def plot_eeg_psd(data, sfreq, ch_names, fmax=60, ax=None, title=None):
    """Plot PSD for multiple EEG channels."""
    from mne.time_frequency import psd_array_welch

    psds, freqs = psd_array_welch(
        data, sfreq=sfreq, fmin=0.5, fmax=fmax,
        n_fft=min(2048, data.shape[1]),
        average='mean'
    )

    psds = 10 * np.log10(psds)  # Convert to dB

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))

    for i, ch in enumerate(ch_names):
        ax.plot(freqs, psds[i], label=ch)

    ax.set_title(title or "EEG Power Spectral Density")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Power Spectral Density (dB/Hz)")
    ax.set_xlim([0, fmax])
    ax.grid(True)
    ax.legend(fontsize='small', ncol=2)

def apply_bandpass_filter(raw, low_cutoff, high_cutoff, method="fir"):
    """
    Apply a bandpass filter to an MNE Raw object.

    Args:
        raw (mne.io.Raw): The EEG data to filter (in-place).
        low_cutoff (float): Low frequency cutoff (Hz).
        high_cutoff (float): High frequency cutoff (Hz).
        method (str): Filtering method ('fir' or 'iir').
    """
    raw.filter(
        l_freq=low_cutoff,
        h_freq=high_cutoff,
        method=method,
        fir_design='firwin',
        fir_window='hamming',
        #l_trans_bandwidth=1.0,  # narrower = sharper filter
        #h_trans_bandwidth=5.0,  # adjust as needed
        phase="zero"
    )
    
    # Apply narrow 50 Hz notch filter
    raw.notch_filter(freqs=50, notch_widths=0.5, trans_bandwidth=0.5, method='fir', fir_design='firwin', fir_window='hamming')


if __name__ == "__main__":
    # Locate result files
    parent_dir = Path(__file__).resolve().parent.parent.parent
    target_dir = os.path.join(parent_dir, "data", "edf", "south_africa", "epileptic_focal_left")

    # Get list of .edf files
    edf_files = [f for f in os.listdir(target_dir) if f.endswith(".edf")]
    if not edf_files:
        raise FileNotFoundError("No EDF files found in target directory.")

    # Pick one randomly
    chosen_file = random.choice(edf_files)
    chosen_path = os.path.join(target_dir, chosen_file)

    # Load and report
    raw = mne.io.read_raw_edf(chosen_path, preload=True)
    sfreq = raw.info['sfreq']
    ch_names = raw.info['ch_names']
    data = raw.get_data()
    times = raw.times

    print(f"Loaded file: {chosen_file}")
    print(f"Shape: {data.shape}")
    print(f"Sampling frequency: {sfreq} Hz")
    print(f"Channels: {ch_names}")

    # Select 20-second segment
    segment_length = int(sfreq * 20)
    max_start = data.shape[1] - segment_length
    if max_start <= 0:
        raise ValueError("Recording too short for a 20-second segment.")

    start = random.randint(0, max_start)
    end = start + segment_length
    segment = data[:, start:end]
    print(f"Selected segment: {start} to {end} ({(end-start)/sfreq}s)")

    # Extract 10–20 standard channels
    segment_raw, filtered_names = extract_21_channels(segment, ch_names)
    if segment_raw.shape[0] == 0:
        raise ValueError("No valid 10–20 channels found in the recording.")

    # Prepare figure
    fig, axs = plt.subplots(2, 2, figsize=(12, 8))
    axs = axs.flatten()

    # Plot raw
    plot_eeg_psd(segment_raw, sfreq, filtered_names, fmax=sfreq / 2, ax=axs[0], title="Unfiltered")

    # Try different high-cut values
    for i, high_cut in enumerate([40.0, 45.0, 49.0]):
        # Clone segment and filter
        info = mne.create_info(filtered_names, sfreq, ch_types="eeg")
        raw_segment = mne.io.RawArray(segment_raw.copy(), info)
        apply_bandpass_filter(raw_segment, low_cutoff=0.5, high_cutoff=high_cut)
        segment_filtered = raw_segment.get_data()

        plot_eeg_psd(segment_filtered, sfreq, filtered_names, fmax=sfreq / 2, ax=axs[i+1], title=f"Filtered (1–{int(high_cut)} Hz)")

    plt.tight_layout()
    plt.show()
