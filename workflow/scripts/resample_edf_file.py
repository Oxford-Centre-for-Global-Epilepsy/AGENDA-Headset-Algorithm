#!/usr/bin/env python3
"""
scripts/resample_data.py

Usage:
    python scripts/resample_data.py <input_file> <output_file>
"""

import sys
import os
import mne 

print("DEBUG: Starting resample_data.py", flush=True)
print("DEBUG: Current working directory:", os.getcwd(), flush=True)

if len(sys.argv) != 4:
    print("Usage: python workflow/resample_data.py <input_file> <output_file> <sampling_frequency>", flush=True)
    sys.exit(1)

input_file = sys.argv[1]
output_file = sys.argv[2]
sampling_frequency = sys.argv[3]

print("DEBUG: Input file:", input_file, flush=True)
print("DEBUG: Output file:", output_file, flush=True)

if not os.path.exists(input_file):
    print("ERROR: Input file does not exist!", flush=True)
    sys.exit(1)

os.makedirs(os.path.dirname(output_file), exist_ok=True)
print("DEBUG: Created/verified output directory:", os.path.dirname(output_file), flush=True)

try:
    
    # Load the input_file (assumes .edf file format)
    raw_data = mne.io.read_raw_edf(input_file, preload=True, verbose=False)

    # Resample the loaded data
    resampled_raw_data = raw_data.copy().resample(sfreq=sampling_frequency)

    # Save the resampled data to the output file
    #mne.export.export_raw(output_file, fmt='edf')
    #resampled_raw_data.export(output_file, fmt='edf')
    resampled_raw_data.save(output_file)

    print("DEBUG: Successfully saved output file", flush=True)
except Exception as e:
    print("ERROR during resample_data.py:", e, flush=True)
    sys.exit(1)
