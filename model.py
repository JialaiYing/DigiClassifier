import torch.nn as nn

class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        # input layer: 784 pixels 
        self.fc1 = nn.Linear(784, 128)
        # reLu activation function
        self.relu = nn.ReLU()
        # hidden layer: 128 neurons, outputs to 64 neurons
        self.fc2 = nn.Linear(128, 64)
        # hidden layer 2: 64 neurons, outputs to 10 neurons
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