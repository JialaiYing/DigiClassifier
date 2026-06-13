import tkinter as tk
import torch
from torchvision import transforms
from PIL import Image, ImageDraw

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