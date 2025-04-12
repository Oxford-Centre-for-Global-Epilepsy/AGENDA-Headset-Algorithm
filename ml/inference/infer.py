import torch
import numpy as np
from ml.models.eegnet import EEGNet

# Load trained model
model = EEGNet(num_classes=2)
model.load_state_dict(torch.load("models/eegnet_model.pth"))
model.eval()

# Load new EEG data
with h5py.File("data/new_patient.h5", "r") as f:
    eeg_data = np.array(f["EEG/data"])

# Run inference
eeg_tensor = torch.tensor(eeg_data, dtype=torch.float32).unsqueeze(1)
with torch.no_grad():
    prediction = model(eeg_tensor)
    label = prediction.argmax(dim=1).item()

print(f"Predicted Label: {label}")
