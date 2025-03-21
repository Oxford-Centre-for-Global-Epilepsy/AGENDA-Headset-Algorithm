#!/usr/bin/env python3
"""
scripts/convert_to_montage.py

Usage:
    python scripts/convert_to_montage.py <input_file> <output_file> <montage_config_path> <montage>
"""

import sys
import os
import mne
import yaml
import numpy as np

def load_montage(montage_path, montage_type, montage_name):
    """Load the correct montage type (raw, monopolar or bipolar)."""
    with open(montage_path, "r") as file:
        montage_config = yaml.safe_load(file)

    # Check if the montage type is specified in the config file
    if montage_type in montage_config["montages"]:

        # Raw Montage -> no montage to be applied to the data
        if montage_type == "raw":
            return "raw", None, None  # No montage applied

        # Bipolar Montage to be applied to the data
        if montage_type == "bipolar" and montage_name in montage_config["montages"]["bipolar"]:
            anodes = np.array(montage_config["montages"]["bipolar"][montage_name]["anodes"])
            cathodes = np.array(montage_config["montages"]["bipolar"][montage_name]["cathodes"])
            return "bipolar", anodes, cathodes

        # Monopolar Montage to be applied to the data
        if montage_type == "monopolar" and montage_name in montage_config["montages"]["monopolar"]:
            channels = montage_config["montages"]["monopolar"][montage_name]["channels"]
            return "monopolar", channels, None

        raise ValueError(f"❌ Unknown montage: {montage_name}")

    # Raise value error for unknown montage
    raise ValueError(f"❌ Unknown montage type: {montage_type}")

def convert_to_bipolar(raw, anodes, cathodes):
    """
    Convert monopolar EEG data to a bipolar montage.

    Parameters:
    raw : mne.io.Raw
        The raw EEG data.
    anodes : np.array
        The anode electrodes for bipolar montage.
    cathodes : np.array
        The cathode electrodes for bipolar montage.

    Returns:
    mne.io.Raw
        The raw EEG data re-referenced to the bipolar montage.
    """
    # Ensure channel names are case-insensitive
    ch_names_lower = {ch.lower(): ch for ch in raw.ch_names}

    bipolar_channel_names = []
    for i in range(anodes.shape[0]):
        for j in range(anodes.shape[1]):
            anode, cathode = anodes[i, j], cathodes[i, j]
            if anode and cathode:
                anode_lower, cathode_lower = anode.lower(), cathode.lower()
                if anode_lower in ch_names_lower and cathode_lower in ch_names_lower:
                    anode_actual = ch_names_lower[anode_lower]
                    cathode_actual = ch_names_lower[cathode_lower]
                    bipolar_name = f"{anode_actual}-{cathode_actual}"

                    raw = mne.set_bipolar_reference(
                        raw,
                        anode_actual,
                        cathode_actual,
                        ch_name=bipolar_name,
                        drop_refs=False,
                        copy=False
                    )
                    bipolar_channel_names.append(bipolar_name)
                else:
                    missing_channels = []
                    if anode_lower not in ch_names_lower:
                        missing_channels.append(anode)
                    if cathode_lower not in ch_names_lower:
                        missing_channels.append(cathode)
                    print(f"⚠️ Warning: Channels {missing_channels} not found in data. Skipping this pair.")

    # Keep only the bipolar channels
    raw.pick_channels(bipolar_channel_names)
    return raw

def convert_to_monopolar(raw, selected_channels):
    """
    Select specific monopolar channels.

    Parameters:
    raw : mne.io.Raw
        The raw EEG data.
    selected_channels : list
        List of channels to retain.

    Returns:
    mne.io.Raw
        The raw EEG data with only selected channels.
    """
    # Ensure case-insensitive matching
    ch_names_lower = {ch.lower(): ch for ch in raw.ch_names}
    selected_channels = [ch_names_lower[ch.lower()] for ch in selected_channels if ch.lower() in ch_names_lower]
    
    raw.pick_channels(selected_channels)
    return raw

def process_montage(input_file, output_file, montage_path, montage_type, montage_name):
    """
    Apply a selected montage to EEG data.

    Parameters:
    input_file : str
        Path to the input EEG file.
    output_file : str
        Path to save the processed EEG file.
    montage_path : str
        Path to the montage configuration YAML file.
    montage_type : str
        Type of montage to apply to the data (i.e. raw, monopolar, bipolar montage).
    montage_name : str
        Name of the montage to apply.
    """
    print(f"🔍 Processing {input_file} with {montage_type} montage: {montage_name}")

    # Load raw EEG data
    raw = mne.io.read_raw_fif(input_file, preload=True, verbose=False)

    # Load montage -
    # For monopolar Montage: param1 = channels of interest
    # For bipolar Montage: param1 = anode channels, param2 = cathode_channels
    montage_type, param1, param2 = load_montage(montage_path, montage_type, montage_name)

    if montage_type == "raw":
        print("⚡ Skipping montage conversion (Raw EEG retained)")
        raw.save(output_file, overwrite=True)
        return

    if montage_type == "monopolar":
        print(f"🔹 Selecting monopolar EEG channels: {param1}")
        raw = convert_to_monopolar(raw, param1)
    
    elif montage_type == "bipolar":
        print(f"⚡ Applying bipolar montage: {montage_name}")
        raw = convert_to_bipolar(raw, param1, param2)

    # Save processed EEG file
    raw.save(output_file, overwrite=True)
    print(f"✅ EEG data saved to {output_file}")

if __name__ == "__main__":
    if len(sys.argv) != 6:
        print("Usage: python scripts/convert_to_montage.py <input_file> <output_file> <montage_config_path> <montage_type> <montage_name>", flush=True)
        sys.exit(1)

    # Parse the arguments from the stdin
    input_file, output_file, montage_path, montage_type, montage_name = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4] , sys.argv[5]
    
    # Process the data with the specified montage
    process_montage(input_file, output_file, montage_path, montage_type, montage_name)
