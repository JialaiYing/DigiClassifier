import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm

# download and load training/testing data
# batches of 64 images
BATCH_SIZE = 64
trainset = torchvision.datasets.MNIST(root='./data', train=True,
                                        download=True, transform=transforms.ToTensor())
testset = torchvision.datasets.MNIST(root='./data', train=False,
                                       download=True, transform=transforms.ToTensor())

train_loader = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False)

# 2. define basic network architecture
class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        # input layer: 784 pixels 
        self.fc1 = nn.Linear(784, 128)
        # reLu activaztion function
        self.relu = nn.ReLU()
        # hidden layer: 128 neurons, outputs to 64 neurons
        self.fc2 = nn.Linear(128, 64)
        # hidden layer 2: 64 neurons, outputs to 10 neruons
        self.fc3 = nn.Linear(64, 10)

    def forward(self, x):
        # transforms image into 1d vector of 784 pixels
        # -1 tells PyTorch to automatically figure out the batch size dimension
        x = x.view(-1, 784)
        
        # Pass through the layers with ReLu activation functions
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x

# initialize network, loss function, and optimizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Network().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# 3. training Loop
EPOCHS = 5
print(f"Training on device: {device}")

for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0
    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
    
    for images, labels in progress_bar:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        progress_bar.set_postfix(loss=running_loss/len(train_loader))

# 4. testing/evaluation Loop
model.eval()
correct = 0
total = 0

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = 100 * correct / total
print(f"\nAccuracy: {accuracy:.2f}%")