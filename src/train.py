import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets
from datetime import datetime
from tqdm import tqdm

from mnist_classifier.model.mnist_network import MNISTModel
from mnist_classifier.utils import (
    compute_accuracy,
    get_train_transforms,
    get_test_transforms,
    TRAIN_CONFIG, 
    DATASET_CONFIG, 
    MODEL_DIR, 
    DATA_DIR
)

def train():
    # Set device
    device = torch.device(TRAIN_CONFIG["device"])
    print(f"Using device: {device}")
    
    # Load MNIST dataset
    transform = get_train_transforms()
    
    print("Downloading/Loading MNIST dataset...")
    train_dataset = datasets.MNIST(DATA_DIR, train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(DATA_DIR, train=False, download=True, transform=get_test_transforms())
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset, 
        batch_size=TRAIN_CONFIG["batch_size"], 
        shuffle=True
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, 
        batch_size=TRAIN_CONFIG["batch_size"], 
        shuffle=False
    )
    
    # Initialize model
    model = MNISTModel().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=TRAIN_CONFIG["learning_rate"])
    
    print("Starting training...")
    model.train()
    pbar = tqdm(train_loader, desc='Training')
    running_loss = 0.0
    correct = 0
    total = 0
    
    for batch_idx, (data, target) in enumerate(pbar):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        # Compute batch accuracy
        _, predicted = torch.max(output.data, 1)
        total += target.size(0)
        correct += (predicted == target).sum().item()
        
        running_loss = 0.9 * running_loss + 0.1 * loss.item()
        train_acc = 100 * correct / total
        
        pbar.set_postfix({
            'loss': f'{running_loss:.4f}',
            'acc': f'{train_acc:.2f}%'
        })
    
    # Compute final metrics
    print("\nComputing final metrics...")
    train_accuracy = compute_accuracy(model, train_loader, device)
    test_accuracy = compute_accuracy(model, test_loader, device)
    
    print(f"\nFinal Results:")
    print(f"Training Accuracy: {train_accuracy:.2f}%")
    print(f"Test Accuracy: {test_accuracy:.2f}%")
    print(f"Final Loss: {running_loss:.4f}")
    
    # Create models directory if it doesn't exist
    MODEL_DIR.mkdir(exist_ok=True)
    
    # Save model with timestamp and accuracy
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    model_filename = f'model_mnist_{timestamp}_acc{test_accuracy:.1f}.pth'
    save_path = MODEL_DIR / model_filename
    
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'test_accuracy': test_accuracy,
        'train_accuracy': train_accuracy,
        'final_loss': running_loss,
        'timestamp': timestamp
    }, save_path)
    
    print(f'Model saved as {save_path}')
    
    # Save a reference to latest model
    latest_model_path = MODEL_DIR / 'latest_model.pth'
    torch.save({
        'model_state_dict': model.state_dict(),
        'test_accuracy': test_accuracy,
    }, latest_model_path)

if __name__ == '__main__':
    train() 