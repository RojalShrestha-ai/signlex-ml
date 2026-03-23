# SignLex ML - Hand Gesture Recognition Pipeline

## Author: Rojal Shrestha (ML/Computer Vision Lead)

## Overview
This module handles the machine learning and computer vision components of SignLex:
- Dataset preprocessing and augmentation pipeline
- MediaPipe Hands integration for 21-point hand landmark detection
- Custom ASL gesture classification model (TensorFlow.js)

## Project Structure
```
rojal/
├── config/
│   └── modelConfig.js       # Model hyperparameters and training config
├── data/
│   ├── preprocess.js         # Dataset cleaning, augmentation, splitting
│   ├── datasetLoader.js      # Load ASL Alphabet and Sign Language MNIST
│   └── augmentation.js       # Data augmentation transforms
├── models/
│   ├── gestureClassifier.js  # Neural network architecture definition
│   ├── train.js              # Model training pipeline (TODO)
│   └── evaluate.js           # Model evaluation metrics (TODO)
├── utils/
│   ├── webcamCapture.js      # Browser webcam access and stream handling
│   ├── handLandmarks.js      # MediaPipe Hands landmark extraction
│   └── visualization.js      # Skeleton overlay rendering (TODO)
└── package.json
```

## Current Status (Report Meeting #1)
- **Goal 1 (Dataset Pipeline): ~30% complete**
  - Dataset loader configured for ASL Alphabet and MNIST
  - Preprocessing pipeline with augmentation transforms defined
  - Train/validation/test split logic implemented (70/15/15)
  
- **Goal 2 (MediaPipe Integration): ~15% complete**
  - Webcam capture utility initialized
  - MediaPipe Hands configuration defined
  - Basic landmark extraction structure in place
  
- **Goal 3 (Model Training): ~10% complete**
  - Model architecture skeleton defined
  - Hyperparameter config file created
  - Training loop and evaluation are TODO

## Setup
```bash
npm install
```

## Datasets Required
1. ASL Alphabet Dataset: https://www.kaggle.com/datasets/grassknoted/asl-alphabet
2. Sign Language MNIST: https://www.kaggle.com/datasets/datamunge/sign-language-mnist

Place downloaded datasets in `data/raw/` directory.
