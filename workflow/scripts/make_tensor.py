import numpy as np
import h5py
import yaml
import sys

def load_montage(montage_path, montage_name):
    """Load spatial montage configuration."""
    with open(montage_path, "r") as file:
        montage_config = yaml.safe_load(file)
    
    anodes = np.array(montage_config["montages"][montage_name]["anodes"])
    cathodes = np.array(montage_config["montages"][montage_name]["cathodes"])

    return anodes, cathodes

def convert_to_tensor(hdf5_input, montage_path, output_tensor):
    """Convert EEG HDF5 file to tensor format with zero-channel placeholders."""
    
    # Load montage and determine grid shape
    montage_name = hdf5_input.split("/")[-1].split("_")[1]  # Extract montage type from filename
    anodes, cathodes = load_montage(montage_path, montage_name)
    grid_shape = anodes.shape  # Get the EEG spatial layout
    
    # Generate a mask indicating valid electrode positions
    valid_positions = np.where(anodes != '', 1, 0)

    # Load EEG data from HDF5
    with h5py.File(hdf5_input, "r") as hf:
        eeg_data = hf["eeg_data"][:]  # Shape: (epochs, channels, timepoints)
    
    num_epochs, num_channels, epoch_length = eeg_data.shape

    # Initialize tensor with zero placeholders
    eeg_tensor = np.zeros((num_epochs, grid_shape[0], grid_shape[1], epoch_length))

    # Fill tensor with EEG data based on valid channels
    channel_idx = 0
    for i in range(grid_shape[0]):
        for j in range(grid_shape[1]):
            if valid_positions[i, j]:  # Assign EEG data only where electrodes exist
                eeg_tensor[:, i, j, :] = eeg_data[:, channel_idx, :]
                channel_idx += 1

    # Save tensor with mask
    with h5py.File(output_tensor, "w") as hf_out:
        hf_out.create_dataset("eeg_tensor", data=eeg_tensor)
        hf_out.create_dataset("valid_positions", data=valid_positions)  # Save mask
    
    print(f"✅ EEG tensor saved at {output_tensor}")

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python make_tensor.py <hdf5_input> <output_tensor> <montage_path>")
        sys.exit(1)

    hdf5_input, output_tensor, montage_path = sys.argv[1], sys.argv[2], sys.argv[3]
    convert_to_tensor(hdf5_input, montage_path, output_tensor)
