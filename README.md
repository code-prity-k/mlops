# MNIST Classification with CI/CD Pipeline

[![ML Pipeline](https://github.com/<your-username>/<repo-name>/actions/workflows/ml_pipeline.yml/badge.svg)](https://github.com/<your-username>/<repo-name>/actions/workflows/ml_pipeline.yml)

A lightweight CNN implementation for MNIST digit classification with automated testing and CI/CD pipeline using GitHub Actions. The model achieves >80% accuracy with less than 50K parameters in just one epoch of training.

## Project Overview

This project implements a lightweight Convolutional Neural Network (CNN) for MNIST digit classification with a focus on efficiency and maintainability. It includes a complete CI/CD pipeline with automated testing and model validation.

## Architecture Details

### Model Structure
- Input Layer: 1 channel (28x28 grayscale images)
- Three 8-channel convolutional blocks for initial feature extraction
- 1x1 convolution to expand to 16 channels
- Two 16-channel convolutional blocks for mid-level features
- 1x1 convolution to expand to 32 channels
- Final 32-channel convolutional block
- Global Average Pooling
- Output Layer: 10 classes (digits 0-9)

### Key Features
- BatchNormalization for training stability
- LeakyReLU activation (slope=0.1)
- Strategic dropout (rate=0.1)
- Global Average Pooling for parameter efficiency
- 1x1 convolutions for channel expansion

### Parameter Count
- Total parameters: ~15,660
- Efficient use of parameters through:
  - Channel progression (8→8→8→16→16→32)
  - 1x1 convolutions
  - Global Average Pooling

## Data Augmentation

The training pipeline includes several augmentation techniques:
- Random rotation (±10 degrees)
- Random translation (±10%)
- Random scaling (90-110%)
- Normalization (mean=0.1307, std=0.3081)

## Project Structure

## Testing Suite

The project includes comprehensive testing:

### Basic Tests
1. Parameter count verification (<50K)
2. Input/output shape validation
3. Model accuracy verification (>80%)

### Advanced Tests
1. Model Robustness Test
   - Verifies consistent predictions under input variations
   - Tests scale invariance properties

2. Augmentation Consistency Test
   - Ensures augmentation preserves image structure
   - Validates dimension consistency
   - Checks contrast preservation

3. Model Confidence Test
   - Evaluates prediction confidence on random noise
   - Prevents overconfident predictions
   - Ensures model calibration

## CI/CD Pipeline

The GitHub Actions pipeline automates:
1. Environment setup
2. Dependency installation
3. Model training
4. Comprehensive testing
5. Model validation

## Model Versioning

Models are saved with detailed information:
- Timestamp
- Test accuracy
- Training metadata
- Format: `model_mnist_YYYYMMDD_HHMMSS_accXX.X.pth`

