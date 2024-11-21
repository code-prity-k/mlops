import pytest
import torch
import os
from pathlib import Path
from mnist_classifier.model.mnist_network import MNISTModel
from mnist_classifier.utils.metrics import count_parameters, compute_accuracy
from mnist_classifier.config import DATASET_CONFIG, MODEL_DIR
from torchvision import datasets, transforms
import numpy as np
from mnist_classifier.utils.transforms import get_train_transforms

def test_model_parameters():
    model = MNISTModel()
    param_count = count_parameters(model)
    assert param_count < 100000, f"Model has {param_count} parameters, should be less than 100000"

def test_input_output_shape():
    model = MNISTModel()
    test_input = torch.randn(1, 1, 28, 28)
    output = model(test_input)
    assert output.shape == (1, 10), f"Output shape is {output.shape}, should be (1, 10)"

def test_model_accuracy():
    device = torch.device('cpu')
    model = MNISTModel().to(device)
    
    # Try to load latest model first
    latest_model_path = MODEL_DIR / 'latest_model.pth'
    if latest_model_path.exists():
        checkpoint = torch.load(latest_model_path)
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        # Fall back to finding the latest model by timestamp
        model_files = list(MODEL_DIR.glob('model_mnist_*.pth'))
        if not model_files:
            pytest.skip("No model file found")
        latest_model = max(model_files)
        checkpoint = torch.load(latest_model)
        model.load_state_dict(checkpoint['model_state_dict'])
    
    # Load test dataset
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(DATASET_CONFIG["mean"], DATASET_CONFIG["std"])
    ])
    test_dataset = datasets.MNIST(
        './data', 
        train=False, 
        download=True, 
        transform=transform
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, 
        batch_size=1000
    )
    
    accuracy = compute_accuracy(model, test_loader, device)
    assert accuracy > 80, f"Model accuracy is {accuracy}%, should be above 80%"

def test_model_robustness():
    """Test model's robustness to input variations"""
    model = MNISTModel()
    model.eval()
    
    # Test with different input scales
    test_input = torch.randn(1, 1, 28, 28)
    base_output = model(test_input)
    scaled_input = test_input * 0.5
    scaled_output = model(scaled_input)
    
    # Check if predictions remain consistent
    _, base_pred = torch.max(base_output, 1)
    _, scaled_pred = torch.max(scaled_output, 1)
    assert base_pred == scaled_pred, "Model predictions should be scale invariant"

def test_augmentation_consistency():
    """Test if augmentation preserves image structure"""
    # Create a simple digit-like pattern
    image = torch.zeros(1, 28, 28)
    image[0, 10:20, 10:20] = 1.0  # Create a square pattern
    
    transform = get_train_transforms()
    augmented = transform(image)
    
    # Check if augmented image maintains basic properties
    assert augmented.shape == (1, 28, 28), "Augmentation should preserve dimensions"
    assert augmented.max() != augmented.min(), "Augmentation should preserve contrast"
    assert not torch.allclose(augmented, image), "Augmentation should modify the image"

def test_model_confidence():
    """Test if model produces reasonable confidence scores"""
    device = torch.device('cpu')
    model = MNISTModel().to(device)
    
    # Load latest model
    latest_model_path = MODEL_DIR / 'latest_model.pth'
    if latest_model_path.exists():
        checkpoint = torch.load(latest_model_path)
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        pytest.skip("No model file found")
    
    # Test with random noise
    noise = torch.randn(1, 1, 28, 28)
    with torch.no_grad():
        output = model(noise)
        probabilities = torch.nn.functional.softmax(output, dim=1)
        max_prob = probabilities.max().item()
        
    # Model should not be too confident on random noise
    assert max_prob < 0.9, f"Model is too confident on random noise: {max_prob:.2f}"

if __name__ == '__main__':
    pytest.main([__file__]) 