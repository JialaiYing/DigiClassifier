import torch
import torch.nn as nn
import torch.optim as optim

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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



# 1. Model with Dropout Only (Dropout rate of 0.2 means 20% of neurons are muted per forward pass)
class NetworkWithDropout(nn.Module):
    def __init__(self, dropout_rate=0.2):
        super(NetworkWithDropout, self).__init__()
        self.fc1 = nn.Linear(784, 128)
        self.relu = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout_rate) # Dropout layer 1
        
        self.fc2 = nn.Linear(128, 64)
        # self.relu shared
        self.dropout2 = nn.Dropout(dropout_rate) # Dropout layer 2
        
        self.fc3 = nn.Linear(64, 10)

    def forward(self, x):
        x = x.view(-1, 784)
        
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout1(x) # Apply dropout after activation
        
        x = self.fc2(x)
        x = self.relu(x)
        x = self.dropout2(x) # Apply dropout after activation
        
        x = self.fc3(x)
        return x

# 2. Model with Both Dropout and L2 Regularization
# (Uses the same architecture as above, but will use a different optimizer setup)
class NetworkWithBoth(nn.Module):
    def __init__(self, dropout_rate=0.2):
        super(NetworkWithBoth, self).__init__()
        self.fc1 = nn.Linear(784, 128)
        self.relu = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout_rate)
        
        self.fc2 = nn.Linear(128, 64)
        self.dropout2 = nn.Dropout(dropout_rate)
        
        self.fc3 = nn.Linear(64, 10)

    def forward(self, x):
        x = x.view(-1, 784)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.dropout2(x)
        x = self.fc3(x)
        return x


# --- Copy 1: L2 Regularization Only --- #
# Uses your original 'Network' architecture. L2 is added via 'weight_decay' in AdamW.
model_l2_only = Network().to(device)
optimizer_l2_only = optim.AdamW(model_l2_only.parameters(), lr=0.01, weight_decay=1e-4)

# --- Copy 2: Dropout Only --- #
# Uses the dropout network structure. Regular AdamW optimizer (no weight decay).
model_dropout_only = NetworkWithDropout(dropout_rate=0.2).to(device)
optimizer_dropout_only = optim.AdamW(model_dropout_only.parameters(), lr=0.01)

# --- Copy 3: Both Dropout and L2 --- #
# Uses the dropout network structure + weight_decay in AdamW.
model_both = NetworkWithBoth(dropout_rate=0.2).to(device)
optimizer_both = optim.AdamW(model_both.parameters(), lr=0.01, weight_decay=1e-4)
