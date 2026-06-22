import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

def get_data_loaders(batch_size=64):
    trainset = torchvision.datasets.MNIST(root='./data', train=True,
                                         download=True, transform=transforms.ToTensor())
    testset = torchvision.datasets.MNIST(root='./data', train=False,
                                        download=True, transform=transforms.ToTensor())

    train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(testset, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader

# Added optimizer parameter with a fallback default
def train_model(model, train_loader, device, epochs=5, optimizer=None):
    criterion = nn.CrossEntropyLoss()
    
    # If no custom optimizer is passed, build the default one
    if optimizer is None:
        optimizer = optim.Adam(model.parameters(), lr=0.01)
    
    print(f"Training on device: {device}")
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        
        for images, labels in progress_bar:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            progress_bar.set_postfix(loss=running_loss/len(train_loader))
            
    # Return the final average training loss for this epoch step
    return running_loss / len(train_loader)

def evaluate_model(model, test_loader, device):
    model.eval()
    criterion = nn.CrossEntropyLoss() # Added to track historical evaluation loss
    correct = 0
    total = 0
    total_loss = 0.0

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            
            # Compute loss value for evaluation tracking
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f"\nAccuracy: {accuracy:.2f}%")
    
    # Return the average testing loss value
    return total_loss / len(test_loader)