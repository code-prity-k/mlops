from torchvision import transforms

def get_train_transforms():
    """Get transforms for training data augmentation"""
    return transforms.Compose([
        transforms.RandomRotation(10),  # Rotate by up to ±10 degrees
        transforms.RandomAffine(
            degrees=0,  # No additional rotation
            translate=(0.1, 0.1),  # Translate by up to 10% in x and y
            scale=(0.9, 1.1)  # Scale by 90-110%
        ),
        transforms.ToTensor(),  # Convert to tensor (0-1 range)
        transforms.Normalize((0.1307,), (0.3081,))  # Normalize with MNIST stats
    ])

def get_test_transforms():
    """Get transforms for test data (no augmentation)"""
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ]) 