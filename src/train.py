import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets
from datetime import datetime
from tqdm import tqdm
import os

from mnist_classifier.model.mnist_network import MNISTModel
from mnist_classifier.utils import (
    compute_accuracy,
    get_train_transforms,
    get_test_transforms,
    count_parameters,
    TRAIN_CONFIG, 
    DATASET_CONFIG, 
    MODEL_DIR, 
    DATA_DIR
)

def train():
    # Set device
    device = torch.device(TRAIN_CONFIG["device"])
    print(f"Using device: {device}")
    
    # Initialize model and show parameter count
    model = MNISTModel().to(device)
    param_count = count_parameters(model)
    print(f"\nModel Parameters: {param_count:,}")
    print(f"Model Size: {param_count * 4 / 1024 / 1024:.2f} MB")
    
    # Load MNIST dataset
    transform = get_train_transforms()
    
    print("\nDownloading/Loading MNIST dataset...")
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
    
    # Print dataset sizes
    print(f"Training samples: {len(train_dataset):,}")
    print(f"Testing samples: {len(test_dataset):,}")
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=TRAIN_CONFIG["learning_rate"])
    
    print("\nStarting training...")
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    # Disable progress bar in CI environment
    disable_tqdm = bool(os.getenv('CI'))
    
    # # Layer-wise parameter count
    # if not disable_tqdm:
    #     print("\nLayer-wise parameters:")
    #     for name, param in model.named_parameters():
    #         if param.requires_grad:
    #             print(f"{name}: {param.numel():,}")
    
    for batch_idx, (data, target) in enumerate(tqdm(train_loader, 
                                                   desc='Training', 
                                                   disable=disable_tqdm, 
                                                   ncols=100)):
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
        
        # Print progress only every 100 batches in non-CI environment
        if not disable_tqdm and batch_idx % 100 == 0:
            train_acc = 100 * correct / total
            print(f'Batch {batch_idx}/{len(train_loader)}, '
                  f'Loss: {running_loss:.4f}, '
                  f'Acc: {train_acc:.2f}%')
    
    # Compute final metrics
    print("\nComputing final metrics...")
    train_accuracy = compute_accuracy(model, train_loader, device)
    test_accuracy = compute_accuracy(model, test_loader, device)
    
    print(f"\nFinal Results:")
    print(f"Model Parameters: {param_count:,}")
    print(f"Training Accuracy: {train_accuracy:.2f}%")
    print(f"Test Accuracy: {test_accuracy:.2f}%")
    print(f"Final Loss: {running_loss:.4f}")
    
    # Save model
    MODEL_DIR.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    model_filename = f'model_mnist_{timestamp}_acc{test_accuracy:.1f}_params{param_count}.pth'
    save_path = MODEL_DIR / model_filename
    
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'test_accuracy': test_accuracy,
        'train_accuracy': train_accuracy,
        'final_loss': running_loss,
        'timestamp': timestamp,
        'parameters': param_count
    }, save_path)
    
    print(f'Model saved as {save_path}')
    
    # Save latest model reference
    latest_model_path = MODEL_DIR / 'latest_model.pth'
    torch.save({
        'model_state_dict': model.state_dict(),
        'test_accuracy': test_accuracy,
        'parameters': param_count
    }, latest_model_path)

if __name__ == '__main__':
    train() 