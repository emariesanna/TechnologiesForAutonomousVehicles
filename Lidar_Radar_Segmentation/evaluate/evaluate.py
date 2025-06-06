# test.py
import torch
from models.pointnet2_seg import PointNet2Seg
from utils.data_utils import TruckScenesDataset
from torch.utils.data import DataLoader

# Parametri
NUM_CLASSES = 10  # Modifica in base al tuo dataset
BATCH_SIZE = 16

# Dataset e DataLoader
test_dataset = TruckScenesDataset(split='test')
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Modello
model = PointNet2Seg(num_classes=NUM_CLASSES)
model.load_state_dict(torch.load('path_to_trained_model.pth'))
model = model.cuda()
model.eval()

# Testing Loop
correct = total = 0
with torch.no_grad():
    for data in test_loader:
        points, labels = data
        points = points.cuda()
        labels = labels.cuda()
        outputs = model(points)
        _, predicted = torch.max(outputs.data, -1)
        total += labels.numel()
        correct += (predicted == labels).sum().item()

print(f"Accuracy: {100 * correct / total}%")
