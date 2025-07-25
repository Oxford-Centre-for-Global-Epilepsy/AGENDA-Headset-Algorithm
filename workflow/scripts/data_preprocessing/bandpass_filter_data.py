import mne
import yaml
import sys
import os

def bandpass_filter(input_file, output_file, config_file):
    """Apply bandpass filter to EEG data."""
    
    # Load YAML config
    with open(config_file, "r") as file:
        config = yaml.safe_load(file)

    # Extract bandpass settings
    low_cutoff = config["bandpass"]["low_cutoff"]
    high_cutoff = config["bandpass"]["high_cutoff"]
    method = config["bandpass"]["method"]  # 'fir' or 'iir'

    print(f"📡 Bandpass filtering {input_file} with {low_cutoff}-{high_cutoff} Hz using {method} method.")

    # Load EEG data
    raw = mne.io.read_raw_edf(input_file, preload=True)

    # Apply bandpass filter
    raw.filter(l_freq=low_cutoff, h_freq=high_cutoff, method=method, fir_design='firwin', fir_window='hamming')

    # Save filtered data
    raw.save(output_file, overwrite=True)
    print(f"✅ Saved filtered EEG to {output_file}")

    # Ensure output file has updated modification time
    try:
        os.utime(output_file, None)
        print(f"DEBUG: Touched output file: {output_file}", flush=True)
    except Exception as e:
        print(f"WARNING: Failed to update mtime: {e}", flush=True)


if __name__ == "__main__":
    input_fif = sys.argv[1]
    output_fif = sys.argv[2]
    config_yaml = sys.argv[3]

    bandpass_filter(input_fif, output_fif, config_yaml)
