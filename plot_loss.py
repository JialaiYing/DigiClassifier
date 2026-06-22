import matplotlib.pyplot as plt

def plot_loss_curves(history):
    """
    Plots the training and testing loss curves for all 4 models.
    'history' should be a dictionary structured like:
    {
        'Model Name': {
            'train_loss': [epoch1_loss, epoch2_loss, ...],
            'test_loss': [epoch1_loss, epoch2_loss, ...]
        }
    }
    """
    epochs = range(1, len(next(iter(history.values()))['train_loss']) + 1)
    
    plt.figure(figsize=(14, 6))
    
    # ---------------------------------------------------------
    # Subplot 1: Training Loss Curve
    # ---------------------------------------------------------
    plt.subplot(1, 2, 1)
    for model_name, metrics in history.items():
        plt.plot(epochs, metrics['train_loss'], marker='o', label=f'{model_name} (Train)')
    
    plt.title('Training Loss Comparison', fontsize=14, fontweight='bold')
    plt.xlabel('Epochs', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.xticks(epochs)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(fontsize=10)
    
    # ---------------------------------------------------------
    # Subplot 2: Testing Loss Curve (The generalization check)
    # ---------------------------------------------------------
    plt.subplot(1, 2, 2)
    for model_name, metrics in history.items():
        plt.plot(epochs, metrics['test_loss'], marker='s', linestyle='--', label=f'{model_name} (Test)')
        
    plt.title('Testing Loss Comparison', fontsize=14, fontweight='bold')
    plt.xlabel('Epochs', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.xticks(epochs)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(fontsize=10)
    
    plt.tight_layout()
    
    # Save the file to your folder and show the visual window
    plt.savefig('model_loss_comparison.png', dpi=300)
    print("\n[Success] Loss curve graph saved to your workspace as 'model_loss_comparison.png'")
    plt.show()