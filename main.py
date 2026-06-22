import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import torch.nn as nn
# ... the rest of your imports and code remain exactly the same ...
# 1. Import the models AND their specific regularized optimizers from model.py
from model import (
    Network,
    model_l2_only, optimizer_l2_only,
    model_dropout_only, optimizer_dropout_only,
    model_both, optimizer_both
)
from train import get_data_loaders, train_model, evaluate_model
from app import AdvancedDrawingApp
from plot_loss import plot_loss_curves       # <-- FIXED: Now imports from plot_loss.py

# Configuration
BATCH_SIZE = 64
EPOCHS = 5

if __name__ == "__main__":
    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Initialize baseline standard model & a dedicated standard optimizer for it
    model_standard = Network().to(device)
    optimizer_standard = torch.optim.Adam(model_standard.parameters(), lr=0.01)
    
    # Load data once (shared across all configurations)
    train_loader, test_loader = get_data_loaders(batch_size=BATCH_SIZE)
    
    # 2. Pair each model with its exact respective optimizer
    models_to_train = [
        ("Standard NN", model_standard, optimizer_standard),
        ("NN with L2 Regularization", model_l2_only, optimizer_l2_only),
        ("NN with Dropout", model_dropout_only, optimizer_dropout_only),
        ("NN with Both", model_both, optimizer_both)
    ]
    
    # Dictionary to keep track of historical training metrics over epochs
    history = {
        "Standard NN": {"train_loss": [], "test_loss": []},
        "NN with L2 Regularization": {"train_loss": [], "test_loss": []},
        "NN with Dropout": {"train_loss": [], "test_loss": []},
        "NN with Both": {"train_loss": [], "test_loss": []}
    }
    
    # 3. Loop through and train every configuration sequentially
    for name, model_instance, optimizer_instance in models_to_train:
        print(f"\n==========================================")
        print(f" Training Model: {name}")
        print(f"==========================================")
        
        for epoch in range(1, EPOCHS + 1):
            # Train the model for exactly 1 epoch at a time to grab its snapshot loss
            avg_train_loss = train_model(model_instance, train_loader, device, epochs=1, optimizer=optimizer_instance) 
            history[name]["train_loss"].append(avg_train_loss)
            
            # Evaluate after the epoch step to get test loss progression
            avg_test_loss = evaluate_model(model_instance, test_loader, device)
            history[name]["test_loss"].append(avg_test_loss)
            
            print(f"Epoch {epoch}/{EPOCHS} -> Train Loss: {avg_train_loss:.4f} | Test Loss: {avg_test_loss:.4f}")

    # --- Generate Loss Curve Visualization ---
    print("\nTraining completed! Generating comparative performance graphs...")
    plot_loss_curves(history)
    
    # 4. Launch front-end drawing interface
    print("\nAll 4 models successfully trained! Launching interface...")
    app = AdvancedDrawingApp()
    app.run()