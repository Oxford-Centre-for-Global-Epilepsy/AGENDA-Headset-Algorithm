import h5py
import os
import torch
from torch.utils.data import Dataset
import numpy as np
from datasets.eeg_dataset import EEGRecordingDataset
from models.eegnet import EEGNet

# Replace this with the path to your newly combined file
DATA_PATH = os.getenv("DATA")
if not DATA_PATH:
    raise ValueError("ERROR: The $DATA environment variable is not set!")

project_folder_path = f"{DATA_PATH}/AGENDA-Headset-Algorithm/"
combined_h5_file = project_folder_path + "data/final_processed/combined_south_africa_monopolar_standard_10_20.h5"

label_map = {
    'neurotypical': 0,
    'epileptic': 1,
    'focal': 2,
    'generalized': 3,
    'left': 4,
    'right': 5
}

dataset = EEGRecordingDataset(
    h5_file_path=combined_h5_file,
    dataset_name="combined_south_africa_monopolar_standard_10_20",
    label_map=label_map
)

# Set up my Dataloader to load my processed EEG data
loader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=True)

# Set up EEGNet
eegnet = EEGNet(num_channels=21, num_samples=129)

# Test on one batch
for batch in loader:
    x = batch["data"]           # shape: [B, E, C, T]
    
    B, E, C, T = x.shape
    print(f"Loaded batch with shape: {x.shape}")
    
    # Reshape for EEGNet: [B*E, 1, C, T]
    x = x.view(B * E, 1, C, T)

    # Forward pass
    out = eegnet(x, return_features=True)
    print("✅ EEGNet ran successfully!")
    print(f"Final feature shape: {out['features'].shape}")
    print(f"Temporal conv output: {out['temporal_features'].shape}")
    print(f"Spatial conv output: {out['spatial_features'].shape}")
    print(f"Separable conv output: {out['final_features'].shape}")
    break
    
    """
    y = batch["labels"]         # shape: [B, 3] (hierarchical labels)
    ids = batch["subject_id"]
    print(y)
    print(ids)
    break
    """

