import torch
from torch import nn, optim
from torchvision import datasets, transforms
import torch.nn.functional as F
from torch.utils.data import DataLoader

from tqdm import tqdm

from model import AlexNet


NUM_EPOCHS = 10
LEARNING_RATE = 0.0001
BATCH_SIZE = 64
TRAIN_TEST_SPLIT = 0.75

device = ("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor()
])

celeba_data = datasets.ImageFolder('data', transform=transform)

train_size = int(len(celeba_data) * TRAIN_TEST_SPLIT)
test_size = len(celeba_data) - train_size

train_set, test_set = torch.utils.data.random_split(
    celeba_data, [train_size, test_size])

train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=True)

model = AlexNet().to(device)

optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
criterion = nn.CrossEntropyLoss()

# Train Loop
for epoch in range(NUM_EPOCHS):
    for batch_idx, (data, labels) in enumerate(tqdm(train_loader)):
        data = data.to(device=device)
        labels = labels.to(device=device)

        # Forward
        predicts = model(data)
        loss = criterion(predicts, labels)

        # Backward
        optimizer.zero_grad()
        loss.backward()

        optimizer.step()

def check_accuracy(loader, model):
    num_correct = 0
    num_samples = 0
    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device=device)
            y = y.to(device=device)
            scores = model(x)
            _, predictions = scores.max(1)
            num_correct += (predictions == y).sum()
            num_samples += predictions.size(0)

    model.train()
    return num_correct/num_samples

print(
    f"Accuracy on training set: {check_accuracy(train_loader, model)*100:.2f}")
print(f"Accuracy on test set: {check_accuracy(test_loader, model)*100:.2f}")
