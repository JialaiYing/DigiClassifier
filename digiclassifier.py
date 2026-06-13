import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
import tkinter as tk
from PIL import Image, ImageDraw, ImageOps

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

class DrawingApp:
    def __init__(self, model, device):
        self.model = model
        self.device = device
        
        # Set up the main GUI window
        self.root = tk.Tk()
        self.root.title("MNIST Digit Predictor")
        
        # Create a canvas for drawing (280x280 scale makes it easy to draw)
        self.canvas_size = 280
        self.canvas = tk.Canvas(self.root, width=self.canvas_size, height=self.canvas_size, bg="black")
        self.canvas.pack(pady=10)
        
        # Create an underlying PIL image to mirror the drawing for processing
        self.image = Image.new("L", (self.canvas_size, self.canvas_size), "black")
        self.draw = ImageDraw.Draw(self.image)
        
        # Label to display predictions
        self.label_result = tk.Label(self.root, text="Draw a digit!", font=("Helvetica", 18))
        self.label_result.pack(pady=10)
        
        # Buttons for clearing and predicting
        self.btn_clear = tk.Button(self.root, text="Clear", command=self.clear_canvas, font=("Helvetica", 12))
        self.btn_clear.pack(side=tk.LEFT, padx=20, pady=10)
        
        self.btn_predict = tk.Button(self.root, text="Predict", command=self.predict_digit, font=("Helvetica", 12))
        self.btn_predict.pack(side=tk.RIGHT, padx=20, pady=10)
        
        # Bind mouse dragging event to drawing function
        self.canvas.bind("<B1-Motion>", self.paint)
        
    def paint(self, event):
        # Draw on both the tkinter canvas and the PIL image
        # Brush radius of 10 creates a thick enough line for MNIST scaling
        r = 10
        x1, y1 = (event.x - r), (event.y - r)
        x2, y2 = (event.x + r), (event.y + r)
        
        self.canvas.create_oval(x1, y1, x2, y2, fill="white", outline="white")
        self.draw.ellipse([x1, y1, x2, y2], fill="white", outline="white")
        
    def clear_canvas(self):
        self.canvas.delete("all")
        self.image = Image.new("L", (self.canvas_size, self.canvas_size), "black")
        self.draw = ImageDraw.Draw(self.image)
        self.label_result.config(text="Draw a digit!")
        
    def predict_digit(self):
        # 1. Resize the image down to 28x28 pixels (MNIST size) using antialiasing
        img_resized = self.image.resize((28, 28), Image.Resampling.LANCZOS)
        
        # 2. Convert PIL image to a PyTorch tensor and normalize to [0, 1]
        # (This matches what transforms.ToTensor() does)
        transform = transforms.ToTensor()
        img_tensor = transform(img_resized).to(self.device)
        
        # 3. Add a batch dimension: [1, 1, 28, 28]
        img_tensor = img_tensor.unsqueeze(0)
        
        # 4. Run model inference
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(img_tensor)
            _, predicted = torch.max(outputs, 1)
            
        # Update the UI with the prediction
        self.label_result.config(text=f"Prediction: {predicted.item()}")
        
    def run(self):
        self.root.mainloop()

# Start the application after training/testing completes
if __name__ == "__main__":
    app = DrawingApp(model, device)
    app.run()