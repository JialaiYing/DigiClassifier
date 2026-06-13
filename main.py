import torch
from model import Network
from train import get_data_loaders, train_model, evaluate_model
from app import DrawingApp

# Configuration
BATCH_SIZE = 64
EPOCHS = 5

if __name__ == "__main__":
    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Initialize network
    model = Network().to(device)
    
    # Load data
    train_loader, test_loader = get_data_loaders(batch_size=BATCH_SIZE)
    
    # Train and Evaluate
    train_model(model, train_loader, device, epochs=EPOCHS) 
    evaluate_model(model, test_loader, device)
    
    # Start the frontend application
    app = DrawingApp(model, device)
    app.run()