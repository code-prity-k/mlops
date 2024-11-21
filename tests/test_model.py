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
    
    # Print layer-wise parameters
    print("\nLayer-wise Parameter Count:")
    total_params = 0
    for name, param in model.named_parameters():
        if param.requires_grad:
            params = param.numel()
            total_params += params
            print(f"{name}: {params:,} parameters")
    
    print(f"\nTotal Parameters: {total_params:,}")
    print(f"Model Size: {total_params * 4 / 1024 / 1024:.2f} MB")
    
    # Calculate percentage of parameters in each layer type
    conv_params = sum(p.numel() for name, p in model.named_parameters() 
                     if 'conv' in name and p.requires_grad)
    bn_params = sum(p.numel() for name, p in model.named_parameters() 
                   if 'bn' in name and p.requires_grad)
    fc_params = sum(p.numel() for name, p in model.named_parameters() 
                   if 'fc' in name and p.requires_grad)
    
    print("\nParameter Distribution:")
    print(f"Convolutional layers: {conv_params:,} ({100*conv_params/total_params:.1f}%)")
    print(f"Batch Normalization: {bn_params:,} ({100*bn_params/total_params:.1f}%)")
    print(f"Fully Connected layers: {fc_params:,} ({100*fc_params/total_params:.1f}%)")
    
    assert param_count < 100000, f"Model has {param_count:,} parameters, should be less than 100,000"

def test_input_output_shape():
    model = MNISTModel()
    test_input = torch.randn(1, 1, 28, 28)
    output = model(test_input)
    
    # Print shape information
    print("\nShape Analysis:")
    print(f"Input shape: {test_input.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Number of classes: {output.shape[1]}")
    
    assert output.shape == (1, 10), f"Output shape is {output.shape}, should be (1, 10)"

def test_model_accuracy():
    device = torch.device('cpu')
    model = MNISTModel().to(device)
    
    # Try to load latest model first
    latest_model_path = MODEL_DIR / 'latest_model.pth'
    if latest_model_path.exists():
        checkpoint = torch.load(latest_model_path, weights_only=True)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"\nLoaded model with {checkpoint.get('parameters', 'unknown')} parameters")
        print(f"Previous test accuracy: {checkpoint.get('test_accuracy', 'unknown'):.2f}%")
    else:
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
    print(f"\nCurrent test accuracy: {accuracy:.2f}%")
    assert accuracy > 80, f"Model accuracy is {accuracy:.2f}%, should be above 80%"

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
    
    # Test with smaller scale variations
    scales = [0.95, 0.975, 1.0, 1.025, 1.05]  # Reduced scale range
    print("\nScale Invariance Test:")
    
    # Create a more structured input
    base_input = torch.zeros(1, 1, 28, 28)
    base_input[0, 0, 10:20, 10:20] = 1.0  # Create a square pattern
    base_input = (base_input - base_input.mean()) / (base_input.std() + 1e-8)
    base_input = base_input * DATASET_CONFIG["std"][0] + DATASET_CONFIG["mean"][0]
    
    with torch.no_grad():
        base_output = model(base_input)
        base_probs = torch.nn.functional.softmax(base_output, dim=1)
        _, base_pred = torch.max(base_output, 1)
        
        print(f"Base prediction confidence: {base_probs.max().item():.2f}")
        max_diff = 0.0
        
        for scale in scales:
            scaled_input = base_input * scale
            scaled_output = model(scaled_input)
            scaled_probs = torch.nn.functional.softmax(scaled_output, dim=1)
            _, scaled_pred = torch.max(scaled_output, 1)
            
            prob_diff = torch.abs(base_probs - scaled_probs).max().item()
            max_diff = max(max_diff, prob_diff)
            print(f"Scale {scale:.3f} - Prediction diff: {prob_diff:.3f}")
        
        # Use maximum difference for assertion
        assert max_diff < 0.3, f"Model predictions vary too much with scale: {max_diff:.3f}"

def test_augmentation_consistency():
    """Test if augmentation preserves image structure"""
    # Create a simple digit-like pattern
    image = np.zeros((28, 28), dtype=np.uint8)
    image[10:20, 10:20] = 255
    
    pil_image = Image.fromarray(image, mode='L')
    transform = get_train_transforms()
    
    print("\nAugmentation Test:")
    print("Testing 5 different augmentations...")
    
    for i in range(5):
        augmented = transform(pil_image)
        print(f"Augmentation {i+1}:")
        print(f"- Shape: {augmented.shape}")
        print(f"- Value range: [{augmented.min():.2f}, {augmented.max():.2f}]")
        
        assert augmented.shape == (1, 28, 28), "Augmentation should preserve dimensions"
        assert augmented.max() != augmented.min(), "Augmentation should preserve contrast"
        assert not torch.allclose(augmented, torch.zeros_like(augmented)), "Augmentation should preserve pattern"

def test_model_confidence():
    """Test if model produces reasonable confidence scores"""
    device = torch.device('cpu')
    model = MNISTModel().to(device)
    
    latest_model_path = MODEL_DIR / 'latest_model.pth'
    if latest_model_path.exists():
        checkpoint = torch.load(latest_model_path, weights_only=True)
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        pytest.skip("No model file found")
    
    print("\nConfidence Test:")
    
    # Test with different noise levels
    noise_levels = [0.1, 0.5, 1.0, 2.0]
    for noise_level in noise_levels:
        noise = torch.randn(1, 1, 28, 28) * noise_level
        with torch.no_grad():
            output = model(noise)
            probabilities = torch.nn.functional.softmax(output, dim=1)
            max_prob = probabilities.max().item()
            print(f"Noise level {noise_level:.1f} - Max confidence: {max_prob:.2f}")
            
        assert max_prob < 0.9, f"Model is too confident on noise (level {noise_level}): {max_prob:.2f}"

if __name__ == '__main__':
    pytest.main([__file__]) 