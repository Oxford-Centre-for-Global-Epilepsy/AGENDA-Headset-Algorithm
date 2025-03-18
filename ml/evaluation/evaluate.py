from sklearn.metrics import accuracy_score, classification_report
import torch

# Load model
model = EEGNet(num_classes=2)
model.load_state_dict(torch.load("models/eegnet_model.pth"))
model.eval()

# Load test data
test_dataset = EEGDataset("data/preprocessed/test.h5")
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Run inference
all_preds = []
all_labels = []
with torch.no_grad():
    for data, labels in test_loader:
        outputs = model(data)
        preds = outputs.argmax(dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

# Compute metrics
accuracy = accuracy_score(all_labels, all_preds)
print(f"Test Accuracy: {accuracy:.4f}")
print(classification_report(all_labels, all_preds))
