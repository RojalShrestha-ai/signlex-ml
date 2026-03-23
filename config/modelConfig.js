const MODEL_CONFIG = {
  // ── Dataset Configuration ──
  datasets: {
    aslAlphabet: {
      name: "ASL Alphabet Dataset",
      source: "https://www.kaggle.com/datasets/grassknoted/asl-alphabet",
      rawPath: "./data/raw/asl_alphabet",
      processedPath: "./data/processed/asl_alphabet",
      numClasses: 29, // 26 letters + space + delete + nothing
      imageSize: 200, // original image dimensions
    },
    signLanguageMNIST: {
      name: "Sign Language MNIST",
      source: "https://www.kaggle.com/datasets/datamunge/sign-language-mnist",
      rawPath: "./data/raw/sign_mnist",
      processedPath: "./data/processed/sign_mnist",
      numClasses: 24, // excludes J and Z (require motion)
      imageSize: 28, // 28x28 grayscale images
    },
  },
  // ── Data Split Ratios ──
  splits: {
    train: 0.70,
    validation: 0.15,
    test: 0.15,
  },
  // ── Augmentation Parameters ──
  augmentation: {
    rotationRange: 15,       // degrees of random rotation
    brightnessRange: 0.2,    // +/- brightness variation factor
    horizontalFlip: true,    // enable horizontal flipping
    zoomRange: 0.1,          // random zoom factor
    shiftRange: 0.1,         // fraction of total dimension for shifting
  },
  // ── MediaPipe Hands Configuration ──
  mediapipe: {
    maxNumHands: 1,
    minDetectionConfidence: 0.7,
    minTrackingConfidence: 0.5,
    modelComplexity: 1,       // 0=lite, 1=full
    numLandmarks: 21,         // 21 hand landmark points
    landmarkDimensions: 3,    // x, y, z per landmark
  },
  // ── Model Architecture ──
  model: {
    inputShape: [784],          // 21 landmarks * 3 coordinates (x, y, z)
    hiddenLayers: [256, 128, 64],    // dense layer units
    dropout: 0.3,              // dropout rate between layers
    outputClasses: 26,         // ASL alphabet A-Z for initial version
    activation: "relu",        // hidden layer activation
    outputActivation: "softmax",
  },
  // ── Training Hyperparameters ──
  training: {
    epochs: 50,
    batchSize: 256,
    learningRate: 0.0005,
    optimizer: "adam",
    lossFunction: "categoricalCrossentropy",
    metrics: ["accuracy"],
    earlyStopping: {
      patience: 10,
      minDelta: 0.001,
      monitor: "val_loss",
    },
    // Initial accuracy target for Report Meeting #1
    targetAccuracy: 0.70,
    // Stretch goal for Report Meeting #2
    stretchAccuracy: 0.85,
  },

  // ── Output Paths ──
  output: {
    modelSavePath: "./models/saved/gesture_model",
    tfjsModelPath: "./models/saved/tfjs_model",
    logsPath: "./models/logs",
    checkpointPath: "./models/checkpoints",
  },
};

// ASL Alphabet label mapping
const ASL_LABELS = {
  0: "A", 1: "B", 2: "C", 3: "D", 4: "E",
  5: "F", 6: "G", 7: "H", 8: "I", 9: "K",
  10: "L", 11: "M", 12: "N", 13: "O", 14: "P",
  15: "Q", 16: "R", 17: "S", 18: "T", 19: "U",
  20: "V", 21: "W", 22: "X", 23: "Y",
  // J (index 9) and Z (index 25) excluded - require motion
};

module.exports = { MODEL_CONFIG, ASL_LABELS };
