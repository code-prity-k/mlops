import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from datetime import datetime
from tqdm import tqdm
import os
import matplotlib.pyplot as plt

from mnist_classifier.model.mnist_network import MNISTModel
from mnist_classifier.utils import (
    compute_accuracy,
    get_train_transforms,
    get_test_transforms,
    count_parameters,
    TRAIN_CONFIG, 
    DATASET_CONFIG, 
    MODEL_DIR, 
    DATA_DIR,
    PROJECT_ROOT
)

def save_image_grid(images, labels, filename, title="Augmented Samples"):
    """Save a grid of images with their labels"""
    plt.figure(figsize=(20, 10))
    for i in range(min(100, len(images))):
        plt.subplot(10, 10, i + 1)
        plt.imshow(images[i].squeeze(), cmap='gray')
        plt.title(f'Label: {labels[i]}')
        plt.axis('off')
    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

def prepare_datasets():
    """Prepare training and test datasets"""
    train_dataset = datasets.MNIST(DATA_DIR, train=True, download=True, transform=get_train_transforms())
    test_dataset = datasets.MNIST(DATA_DIR, train=False, download=True, transform=get_test_transforms())
    
    return train_dataset, test_dataset

def create_data_loaders(train_dataset, test_dataset):
    """Create data loaders for training and testing"""
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
    return train_loader, test_loader

def visualize_augmentations(train_dataset):
    """Visualize augmented images"""
    viz_dir = PROJECT_ROOT / "visualizations"
    viz_dir.mkdir(exist_ok=True)
    
    # Get original images without augmentation
    orig_dataset = datasets.MNIST(DATA_DIR, train=True, download=True,
                                transform=transforms.ToTensor())
    
    orig_images = []
    aug_images = []
    labels = []
    
    for i in range(100):
        orig_img, label = orig_dataset[i]
        aug_img, _ = train_dataset[i]
        
        orig_images.append(orig_img)
        aug_images.append(aug_img)
        labels.append(label)
    
    # Save visualization grids
    save_image_grid(orig_images, labels, viz_dir / "original_samples.png", "Original Samples")
    save_image_grid(aug_images, labels, viz_dir / "augmented_samples.png", "Augmented Samples")
    
    return viz_dir

def save_model(model, optimizer, train_accuracy, test_accuracy, running_loss, param_count):
    """Save model checkpoints"""
    MODEL_DIR.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Save full model checkpoint
    model_filename = f'model_mnist_{timestamp}_acc{test_accuracy:.1f}_params{param_count}.pth'
    save_path = MODEL_DIR / model_filename
    
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'test_accuracy': test_accuracy,
        'train_accuracy': train_accuracy,
        'final_loss': running_loss,
        'timestamp': timestamp,
        'parameters': param_count
    }
    torch.save(checkpoint, save_path)
    
    # Save latest model reference
    latest_path = MODEL_DIR / 'latest_model.pth'
    latest_checkpoint = {
        'model_state_dict': model.state_dict(),
        'test_accuracy': test_accuracy,
        'parameters': param_count
    }
    torch.save(latest_checkpoint, latest_path)
    
    return save_path

def train_epoch(model, train_loader, criterion, optimizer, device, is_ci=False):
    """Train for one epoch"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(train_loader, desc='Training', ncols=100, leave=True)
    
    for batch_idx, (data, target) in enumerate(pbar):
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        # Update metrics
        _, predicted = torch.max(output.data, 1)
        total += target.size(0)
        correct += (predicted == target).sum().item()
        running_loss = 0.9 * running_loss + 0.1 * loss.item()
        train_acc = 100 * correct / total
        
        # Update progress bar
        pbar.set_description(
            f'Batch {batch_idx}/{len(train_loader)}, '
            f'Loss: {running_loss:.4f}, '
            f'Acc: {train_acc:.2f}%'
        )
        
        if is_ci and (batch_idx + 1) % (len(train_loader) // 10) == 0:
            print(f"Progress: {100*(batch_idx+1)/len(train_loader):.1f}%, "
                  f"Loss: {running_loss:.4f}, "
                  f"Acc: {train_acc:.2f}%")
    
    pbar.close()
    return running_loss, train_acc

def train():
    """Main training function"""
    is_ci = bool(os.getenv('CI'))
    device = torch.device(TRAIN_CONFIG["device"])
    print(f"Using device: {device}")
    
    # Initialize model
    model = MNISTModel().to(device)
    param_count = count_parameters(model)
    print(f"\nModel Parameters: {param_count:,}")
    print(f"Model Size: {param_count * 4 / 1024 / 1024:.2f} MB")
    
    # Prepare data
    train_dataset, test_dataset = prepare_datasets()
    if not is_ci:
        viz_dir = visualize_augmentations(train_dataset)
        print(f"\nVisualization saved to {viz_dir}")
    
    train_loader, test_loader = create_data_loaders(train_dataset, test_dataset)
    print(f"\nTraining samples: {len(train_dataset):,}")
    print(f"Testing samples: {len(test_dataset):,}")
    
    # Training setup
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=TRAIN_CONFIG["learning_rate"])
    
    # Train model
    print("\nStarting training...")
    running_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device, is_ci)
    
    # Evaluate model
    print("\nComputing final metrics...")
    train_accuracy = compute_accuracy(model, train_loader, device)
    test_accuracy = compute_accuracy(model, test_loader, device)
    
    print(f"\nFinal Results:")
    print(f"Model Parameters: {param_count:,}")
    print(f"Training Accuracy: {train_accuracy:.2f}%")
    print(f"Test Accuracy: {test_accuracy:.2f}%")
    print(f"Final Loss: {running_loss:.4f}")
    
    # Save model
    save_path = save_model(model, optimizer, train_accuracy, test_accuracy, running_loss, param_count)
    print(f'Model saved as {save_path}')

if __name__ == '__main__':
    train() 