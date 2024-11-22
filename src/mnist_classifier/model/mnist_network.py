import torch
import torch.nn as nn
from mnist_classifier.utils.config import MODEL_CONFIG

class MNISTModel(nn.Module):
    def __init__(self):
        super(MNISTModel, self).__init__()
        
        dropout_rate = MODEL_CONFIG["dropout_rate"]
        
        # First block: Initial feature extraction (8)
        self.features = nn.Sequential(
            # Initial 8-channel blocks
            nn.Conv2d(1, 8, kernel_size=3, padding=1),
            nn.BatchNorm2d(8),
            nn.LeakyReLU(0.1),
            
            nn.Conv2d(8, 8, kernel_size=3, padding=1),
            nn.BatchNorm2d(8),
            nn.LeakyReLU(0.1),
            
            nn.Conv2d(8, 8, kernel_size=3, padding=1),
            nn.BatchNorm2d(8),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(2, 2),
            nn.Dropout(dropout_rate),
            
            # First 1x1 expansion (8->16)
            nn.Conv2d(8, 16, kernel_size=1),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.1),
            
            # 16-channel blocks
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.1),
            
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(2, 2),
            nn.Dropout(dropout_rate),
            
            # Second 1x1 expansion (16->32)
            nn.Conv2d(16, 32, kernel_size=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.1),
            
            # Final 32-channel block
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(2, 2),
            nn.Dropout(dropout_rate),
            
            # Global pooling
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten()
        )
        
        # Final classifier
        self.classifier = nn.Linear(32, MODEL_CONFIG["fc_features"])
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize model weights for better training"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x