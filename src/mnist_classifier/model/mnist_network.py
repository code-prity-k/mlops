import torch
import torch.nn as nn
from mnist_classifier.utils.config import MODEL_CONFIG

class MNISTModel(nn.Module):
    def __init__(self):
        super(MNISTModel, self).__init__()
        
        dropout_rate = MODEL_CONFIG["dropout_rate"]
        
        # First block: Initial feature extraction (8)
        self.conv1 = nn.Conv2d(1, 8, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(8)
        
        # Second block: Maintain spatial info (8)
        self.conv2 = nn.Conv2d(8, 8, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(8)
        
        # Third block: More spatial features (8)
        self.conv3 = nn.Conv2d(8, 8, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(8)
        
        # Feature expansion with 1x1 (8->16)
        self.conv1x1_1 = nn.Conv2d(8, 16, kernel_size=1)
        self.bn1x1_1 = nn.BatchNorm2d(16)
        
        # Fourth block: Higher features (16)
        self.conv4 = nn.Conv2d(16, 16, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(16)
        
        # Fifth block: More complex features (16)
        self.conv5 = nn.Conv2d(16, 16, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm2d(16)
        
        # Feature expansion with 1x1 (16->32)
        self.conv1x1_2 = nn.Conv2d(16, 32, kernel_size=1)
        self.bn1x1_2 = nn.BatchNorm2d(32)
        
        # Final block: Complex features (32)
        self.conv6 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.bn6 = nn.BatchNorm2d(32)
        
        # Global Average Pooling
        self.gap = nn.AdaptiveAvgPool2d(1)
        
        # Final classifier
        self.fc = nn.Linear(32, MODEL_CONFIG["fc_features"])
        
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(dropout_rate)
        self.relu = nn.LeakyReLU(0.1)
        
    def forward(self, x):
        # First three blocks (8 channels)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.pool(x)  # 28x28 -> 14x14
        x = self.dropout(x)
        
        # First 1x1 expansion (8->16)
        x = self.conv1x1_1(x)
        x = self.bn1x1_1(x)
        x = self.relu(x)
        
        # Two 16-channel blocks
        x = self.conv4(x)
        x = self.bn4(x)
        x = self.relu(x)
        
        x = self.conv5(x)
        x = self.bn5(x)
        x = self.relu(x)
        x = self.pool(x)  # 14x14 -> 7x7
        x = self.dropout(x)
        
        # Second 1x1 expansion (16->32)
        x = self.conv1x1_2(x)
        x = self.bn1x1_2(x)
        x = self.relu(x)
        
        # Final 32-channel block
        x = self.conv6(x)
        x = self.bn6(x)
        x = self.relu(x)
        x = self.pool(x)  # 7x7 -> 3x3
        x = self.dropout(x)
        
        # Global pooling and classification
        x = self.gap(x)
        x = x.view(-1, 32)
        x = self.fc(x)
        return x