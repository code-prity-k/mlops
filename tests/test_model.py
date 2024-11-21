import pytest
import torch
import os
import numpy as np
from pathlib import Path
from mnist_classifier.model.mnist_network import MNISTModel
from mnist_classifier.utils.metrics import count_parameters, compute_accuracy
from mnist_classifier.utils.config import DATASET_CONFIG, MODEL_DIR
from mnist_classifier.utils.transforms import get_train_transforms
from torchvision import datasets, transforms
from PIL import Image

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
        checkpoint = torch.load(latest_model_path, weights_only=True)
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        # Fall back to finding the latest model by timestamp
        model_files = list(MODEL_DIR.glob('model_mnist_*.pth'))
        if not model_files:
            pytest.skip("No model file found")
        latest_model = max(model_files)
        checkpoint = torch.load(latest_model, weights_only=True)
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
    device = torch.device('cpu')
    model = MNISTModel().to(device)
    
    # Load the trained model
    latest_model_path = MODEL_DIR / 'latest_model.pth'
    if latest_model_path.exists():
        checkpoint = torch.load(latest_model_path, weights_only=True)
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        pytest.skip("No model file found")
    
    model.eval()
    
    # Create a more realistic test input using MNIST mean and std
    test_input = torch.randn(1, 1, 28, 28)
    test_input = (test_input - test_input.mean()) / test_input.std()
    test_input = test_input * DATASET_CONFIG["std"][0] + DATASET_CONFIG["mean"][0]
    
    with torch.no_grad():
        base_output = model(test_input)
        # Test with slight scale variation
        scaled_input = test_input * 0.95  # Reduced scale difference
        scaled_output = model(scaled_input)
        
        # Compare softmax probabilities instead of hard predictions
        base_probs = torch.nn.functional.softmax(base_output, dim=1)
        scaled_probs = torch.nn.functional.softmax(scaled_output, dim=1)
        
        # Check if probability distributions are similar
        prob_diff = torch.abs(base_probs - scaled_probs).max().item()
        assert prob_diff < 0.1, "Model predictions should be relatively scale invariant"

def test_augmentation_consistency():
    """Test if augmentation preserves image structure"""
    # Create a simple digit-like pattern as a numpy array
    image = np.zeros((28, 28), dtype=np.uint8)
    image[10:20, 10:20] = 255  # Create a square pattern
    
    # Convert to PIL Image
    pil_image = Image.fromarray(image, mode='L')
    
    transform = get_train_transforms()
    augmented = transform(pil_image)
    
    # Check if augmented image maintains basic properties
    assert augmented.shape == (1, 28, 28), "Augmentation should preserve dimensions"
    assert augmented.max() != augmented.min(), "Augmentation should preserve contrast"
    assert not torch.allclose(augmented, torch.zeros_like(augmented)), "Augmentation should preserve pattern"

def test_model_confidence():
    """Test if model produces reasonable confidence scores"""
    device = torch.device('cpu')
    model = MNISTModel().to(device)
    
    # Load latest model
    latest_model_path = MODEL_DIR / 'latest_model.pth'
    if latest_model_path.exists():
        checkpoint = torch.load(latest_model_path, weights_only=True)
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