#!/usr/bin/env python3
"""
Segment EEG data into fixed-length overlapping windows.
Usage:
    python scripts/epoch_data.py <input_file> <output_file> <config_file>
"""

import sys
import os
import mne
import yaml
import numpy as np

def load_config(config_file):
    """Load YAML configuration file."""
    with open(config_file, "r") as file:
        return yaml.safe_load(file)

def window_data(input_fif, output_fif, config_file):
    """Splits EEG data into overlapping windows (epochs)."""
    print("🔄 Loading config file...")
    config = load_config(config_file)

    # Extract parameters
    window_duration = config.get("window_duration", 2.0)  # Seconds
    overlap = config.get("overlap", 0.5)  # Percentage
    baseline = config.get("baseline", (None, 0))

    # Convert string "None" to actual None
    if isinstance(baseline[0], str) and baseline[0].lower() == "none":
        baseline[0] = None
    if isinstance(baseline[1], str) and baseline[1].lower() == "none":
        baseline[1] = None

    print(f"📂 Loading EEG data from: {input_fif}")
    raw = mne.io.read_raw_fif(input_fif, preload=True)

    sfreq = raw.info["sfreq"]  # Sampling frequency
    window_samples = int(window_duration * sfreq)  # Convert duration to samples
    step_size = int(window_samples * (1 - overlap))  # Step size for sliding windows

    print(f"🧩 Splitting data into {window_duration}s windows with {overlap*100:.0f}% overlap...")

    # Generate window start times
    start_samples = np.arange(0, raw.n_times - window_samples + 1, step_size)
    
    # Create fake events to define window boundaries
    events = np.array([[start, 0, 1] for start in start_samples])

    # Adjust tmax to get exactly `window_samples` samples per epoch
    adjusted_tmax = (window_samples - 1) / sfreq

    epochs = mne.Epochs(
        raw, events, event_id=1, tmin=0, tmax=adjusted_tmax, baseline=baseline,
        detrend=1, preload=True
    )


    print(f"💾 Saving windowed data to: {output_fif}")
    epochs.save(output_fif, overwrite=True)

    print("✅ Windowing complete!")

    # Ensure output file has updated modification time
    try:
        os.utime(output_file, None)
        print(f"DEBUG: Touched output file: {output_file}", flush=True)
    except Exception as e:
        print(f"WARNING: Failed to update mtime: {e}", flush=True)


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python scripts/window_data.py <input_file> <output_file> <config_file>")
        sys.exit(1)

    input_fif = sys.argv[1]
    output_fif = sys.argv[2]
    config_file = sys.argv[3]

    if not os.path.exists(input_fif):
        print(f"❌ ERROR: Input file {input_fif} does not exist!")
        sys.exit(1)

    os.makedirs(os.path.dirname(output_fif), exist_ok=True)

    try:
        window_data(input_fif, output_fif, config_file)
    except Exception as e:
        print(f"❌ ERROR during windowing: {e}")
        sys.exit(1)
