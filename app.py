import tkinter as tk
from tkinter import ttk
import torch
from torchvision import transforms
from PIL import Image, ImageDraw

# Import your models and device setup directly from your model.py file
from model import (
    device,
    Network,
    NetworkWithDropout,
    NetworkWithBoth,
    model_l2_only,
    model_dropout_only,
    model_both
)

# Initialize your original standard model right here (since model.py holds the other 3)
model_standard = Network().to(device)

# Map human-readable dropdown options to the actual model instances
MODEL_MAP = {
    "Standard NN": model_standard,
    "NN with L2 Regularization": model_l2_only,
    "NN with Dropout": model_dropout_only,
    "NN with Both (L2 + Dropout)": model_both
}

class AdvancedDrawingApp:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("MNIST Model Selector & Predictor")
        
        # --- Model Selector Layout ---
        self.frame_top = tk.Frame(self.root)
        self.frame_top.pack(pady=10, padx=10)
        
        self.lbl_select = tk.Label(self.frame_top, text="Select Model:", font=("Helvetica", 11))
        self.lbl_select.pack(side=tk.LEFT, padx=5)
        
        # Dropdown menu for selecting the model configuration
        self.model_selector = ttk.Combobox(
            self.frame_top, 
            values=list(MODEL_MAP.keys()), 
            state="readonly", 
            width=25,
            font=("Helvetica", 11)
        )
        self.model_selector.set("Standard NN") # Default option
        self.model_selector.pack(side=tk.LEFT, padx=5)
        
        # --- Drawing Canvas ---
        self.canvas_size = 280
        self.canvas = tk.Canvas(self.root, width=self.canvas_size, height=self.canvas_size, bg="black")
        self.canvas.pack(pady=10)
        
        # Mirror PIL image for downsampling processing
        self.image = Image.new("L", (self.canvas_size, self.canvas_size), "black")
        self.draw = ImageDraw.Draw(self.image)
        
        # Result Display Labels
        self.label_result = tk.Label(self.root, text="Draw a digit!", font=("Helvetica", 18, "bold"))
        self.label_result.pack(pady=5)
        
        self.label_subtext = tk.Label(self.root, text="Using: Standard NN", font=("Helvetica", 10), fg="gray")
        self.label_subtext.pack(pady=5)
        
        # --- Control Buttons ---
        self.frame_buttons = tk.Frame(self.root)
        self.frame_buttons.pack(pady=10)
        
        self.btn_clear = tk.Button(self.frame_buttons, text="Clear", command=self.clear_canvas, font=("Helvetica", 11), width=10)
        self.btn_clear.pack(side=tk.LEFT, padx=15)
        
        self.btn_predict = tk.Button(self.frame_buttons, text="Predict", command=self.predict_digit, font=("Helvetica", 11), width=10, bg="#2196F3", fg="white")
        self.btn_predict.pack(side=tk.RIGHT, padx=15)
        
        # Bind drawing mechanism
        self.canvas.bind("<B1-Motion>", self.paint)
        # Update subtext status when selection changes
        self.model_selector.bind("<<ComboboxSelected>>", self.update_status_text)
        
        # Force window to focus on top of workspace
        self.root.lift()
        self.root.attributes('-topmost', True)
        self.root.after_idle(self.root.attributes, '-topmost', False)

    def paint(self, event):
        r = 10  # Optimal brush thickness for 28x28 downsizing
        x1, y1 = (event.x - r), (event.y - r)
        x2, y2 = (event.x + r), (event.y + r)
        self.canvas.create_oval(x1, y1, x2, y2, fill="white", outline="white")
        self.draw.ellipse([x1, y1, x2, y2], fill="white", outline="white")
        
    def clear_canvas(self):
        self.canvas.delete("all")
        self.image = Image.new("L", (self.canvas_size, self.canvas_size), "black")
        self.draw = ImageDraw.Draw(self.image)
        self.label_result.config(text="Draw a digit!")
        
    def update_status_text(self, event):
        selected_name = self.model_selector.get()
        self.label_subtext.config(text=f"Using: {selected_name}")

    def predict_digit(self):
        # 1. Grab the currently selected model string and map it to the live network
        selected_model_name = self.model_selector.get()
        active_model = MODEL_MAP[selected_model_name]
        
        # 2. Resize drawn image down to MNIST size (28x28)
        img_resized = self.image.resize((28, 28), Image.Resampling.LANCZOS)
        
        # 3. Transform to Tensor and push to CPU/CUDA device
        transform = transforms.ToTensor()
        img_tensor = transform(img_resized).to(device)
        img_tensor = img_tensor.unsqueeze(0)  # Shape becomes [1, 1, 28, 28]
        
        # 4. Process inference safely
        active_model.eval()
        with torch.no_grad():
            outputs = active_model(img_tensor)
            _, predicted = torch.max(outputs, 1)
            
        # UI Update
        self.label_result.config(text=f"Prediction: {predicted.item()}")

    def run(self):
        self.root.mainloop()

if __name__ == "__main__":
    app = AdvancedDrawingApp()
    app.run()