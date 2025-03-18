import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from datasets.eeg_dataset import EEGDataset
from models.eegnet import EEGNet

# Load dataset
train_dataset = EEGDataset("data/preprocessed/train.h5")
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# Initialize model
model = EEGNet(num_classes=2)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

# Training loop
for epoch in range(10):
    model.train()
    running_loss = 0.0
    for data, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(data)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    
    print(f"Epoch {epoch+1}, Loss: {running_loss:.4f}")

# Save trained model
torch.save(model.state_dict(), "models/eegnet_model.pth")
