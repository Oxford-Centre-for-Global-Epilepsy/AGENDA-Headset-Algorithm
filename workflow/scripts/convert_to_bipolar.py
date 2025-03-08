#!/usr/bin/env python3
"""
scripts/convert_to_bipolar.py

Usage:
    python scripts/convert_to_bipolar.py <input_file> <output_file> <montage_config_path> <montage>
"""

import sys
import os
import mne 
import yaml

def load_montage_config(config_path, montage_type):
    """Load montage configuration from a YAML file."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config["montages"][montage_type]

def convert_to_bipolar(raw, spatial_config):
    """
    Convert monopolar EEG data to a bipolar montage based on the provided spatial configuration.

    Parameters:
    raw : mne.io.Raw
        The raw EEG data in a monopolar montage.
    spatial_config : dict
        A dictionary containing 'anodes' and 'cathodes' lists that define the bipolar montage.

    Returns:
    mne.io.Raw
        The raw EEG data re-referenced to the bipolar montage.
    """
    anodes = spatial_config['anodes']
    cathodes = spatial_config['cathodes']

    # Flatten the lists and remove empty entries
    anodes_flat = [ch for sublist in anodes for ch in sublist if ch]
    cathodes_flat = [ch for sublist in cathodes for ch in sublist if ch]

    # Ensure channel names are case-insensitive
    ch_names_lower = {ch_name.lower(): ch_name for ch_name in raw.ch_names}

    # Initialize lists to keep track of channels to drop later
    channels_to_drop = []

    for anode, cathode in zip(anodes_flat, cathodes_flat):
        anode_lower = anode.lower()
        cathode_lower = cathode.lower()

        if anode_lower in ch_names_lower and cathode_lower in ch_names_lower:
            anode_actual = ch_names_lower[anode_lower]
            cathode_actual = ch_names_lower[cathode_lower]
            bipolar_name = f"{anode_actual}-{cathode_actual}"

            # Create bipolar channel
            raw = mne.set_bipolar_reference(
                raw,
                anode_actual,
                cathode_actual,
                ch_name=bipolar_name,
                drop_refs=False,
                copy=False
            )

            # Add original channels to the drop list
            channels_to_drop.extend([anode_actual, cathode_actual])
        else:
            missing_channels = []
            if anode_lower not in ch_names_lower:
                missing_channels.append(anode)
            if cathode_lower not in ch_names_lower:
                missing_channels.append(cathode)
            print(f"Warning: Channels {missing_channels} not found in data. Skipping this pair.")

    # Drop original monopolar channels
    raw.drop_channels(channels_to_drop)

    return raw

print("DEBUG: Starting convert_to_bipolar.py", flush=True)
print("DEBUG: Current working directory:", os.getcwd(), flush=True)

if len(sys.argv) != 5:
    print("Usage: python workflow/convert_to_bipolar.py <input_file> <output_file> <montage_config_path> <montage>", flush=True)
    sys.exit(1)

input_file = sys.argv[1]
output_file = sys.argv[2]
montage_config_path = sys.argv[3]
montage_type = sys.argv[4]

print("DEBUG: Input file:", input_file, flush=True)
print("DEBUG: Output file:", output_file, flush=True)
print("DEBUG: Montage Configuration File:", montage_config_path, flush=True)
print("DEBUG: Montage:", montage_type, flush=True)

if not os.path.exists(input_file):
    print("ERROR: Input file does not exist!", flush=True)
    sys.exit(1)

os.makedirs(os.path.dirname(output_file), exist_ok=True)
print("DEBUG: Created/verified output directory:", os.path.dirname(output_file), flush=True)

try:

    # Load the input_file (assumes .fif file format)
    raw_data = mne.io.read_raw_fif(input_file, preload=True, verbose=False)

    # Load the montage configuration the data should be converted to
    montage_config = load_montage_config(montage_config_path, montage_type)

    # Convert the data to the specified bipolar montage
    bipolar_raw_data = convert_to_bipolar(raw=raw_data, spatial_config=montage_config)

    # Save the bipolar raw data to the output file
    bipolar_raw_data.save(output_file)

    print("DEBUG: Successfully saved output file", flush=True)
except Exception as e:
    print("ERROR during convert_to_bipolar.py:", e, flush=True)
    sys.exit(1)
