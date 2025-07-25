#!/usr/bin/env python3
"""
Resamples EEG data to a specified sampling frequency.

Usage:
    python scripts/resample_data.py <input_file> <output_file> <config_file>
"""

import sys
import os
import mne
import yaml

# 🚀 Debugging info
print("DEBUG: Starting resample_data.py", flush=True)
print(f"DEBUG: Current working directory: {os.getcwd()}", flush=True)

# ✅ Check command-line arguments
if len(sys.argv) != 4:
    print("Usage: python scripts/resample_data.py <input_file> <output_file> <config_file>", flush=True)
    sys.exit(1)

input_file = sys.argv[1]
output_file = sys.argv[2]
config_file = sys.argv[3]

# ✅ Load YAML config
try:
    with open(config_file, "r") as file:
        config = yaml.safe_load(file)
except Exception as e:
    print(f"ERROR: Failed to load config file {config_file}: {e}", flush=True)
    sys.exit(1)

# ✅ Get sampling frequency from config
sampling_frequency = config.get("sampling_frequency")
if sampling_frequency is None:
    print("ERROR: 'sampling_frequency' not found in config file!", flush=True)
    sys.exit(1)

# ✅ Debug info
print(f"DEBUG: Input file: {input_file}", flush=True)
print(f"DEBUG: Output file: {output_file}", flush=True)
print(f"DEBUG: Target sampling frequency: {sampling_frequency} Hz", flush=True)

# ✅ Check input file exists
if not os.path.exists(input_file):
    print(f"ERROR: Input file {input_file} does not exist!", flush=True)
    sys.exit(1)

# ✅ Ensure output directory exists
output_dir = os.path.dirname(output_file)
os.makedirs(output_dir, exist_ok=True)
print(f"DEBUG: Created/verified output directory: {output_dir}", flush=True)

try:
    # ✅ Determine input file format dynamically
    if input_file.endswith(".edf"):
        print("DEBUG: Detected EDF file format", flush=True)
        raw_data = mne.io.read_raw_edf(input_file, preload=True, verbose=False)
    elif input_file.endswith(".fif"):
        print("DEBUG: Detected FIF file format", flush=True)
        raw_data = mne.io.read_raw_fif(input_file, preload=True, verbose=False)
    else:
        print("ERROR: Unsupported file format! Only .edf and .fif are supported.", flush=True)
        sys.exit(1)

    # ✅ Apply resampling
    print("DEBUG: Resampling data...", flush=True)
    resampled_raw_data = raw_data.copy().resample(sfreq=sampling_frequency)

    # ✅ Save resampled data
    resampled_raw_data.save(output_file, overwrite=True)
    print(f"✅ Successfully saved resampled data to {output_file}", flush=True)

except Exception as e:
    print(f"ERROR during resample_data.py: {e}", flush=True)
    sys.exit(1)

# Ensure output file has updated modification time
try:
    os.utime(output_file, None)
    print(f"DEBUG: Touched output file: {output_file}", flush=True)
except Exception as e:
    print(f"WARNING: Failed to update mtime: {e}", flush=True)
