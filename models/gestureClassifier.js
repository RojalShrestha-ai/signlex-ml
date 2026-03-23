const { MODEL_CONFIG } = require("../config/modelConfig");

/**
 * Build a CNN gesture classification model.
 * Much better for image data than dense networks.
 * 
 * Architecture:
 *   Input: [784] (flattened 28x28 grayscale)
 *   Reshape → [28, 28, 1]
 *   Conv Block 1: Conv2D(32) → BN → Conv2D(32) → MaxPool → Dropout(0.25)
 *   Conv Block 2: Conv2D(64) → BN → Conv2D(64) → MaxPool → Dropout(0.25)
 *   Conv Block 3: Conv2D(128) → BN → Conv2D(128) → MaxPool → Dropout(0.25)
 *   Dense Head: Flatten → Dense(256) → BN → Dropout(0.5) → Dense(128) → Dropout(0.3)
 *   Output: Dense(26, softmax)
 *
 * @param {Object} tf - TensorFlow.js module
 * @returns {tf.Sequential} compiled model
 */
function buildGestureModel(tf) {
  const model = tf.sequential({ name: "signlex-cnn-classifier" });

  // Reshape flattened 784 input back to 28x28x1 image
  model.add(
    tf.layers.reshape({
      inputShape: [784],
      targetShape: [28, 28, 1],
      name: "reshape_input",
    })
  );

  // ── Conv Block 1 ──
  model.add(
    tf.layers.conv2d({
      filters: 32,
      kernelSize: 3,
      activation: "relu",
      padding: "same",
      kernelInitializer: "heNormal",
      name: "conv1a",
    })
  );
  model.add(
    tf.layers.batchNormalization({ name: "bn1a" })
  );
  model.add(
    tf.layers.conv2d({
      filters: 32,
      kernelSize: 3,
      activation: "relu",
      padding: "same",
      kernelInitializer: "heNormal",
      name: "conv1b",
    })
  );
  model.add(
    tf.layers.maxPooling2d({
      poolSize: 2,
      strides: 2,
      name: "pool1",
    })
  );
  model.add(
    tf.layers.dropout({
      rate: 0.25,
      name: "dropout1",
    })
  );

  // ── Conv Block 2 ──
  model.add(
    tf.layers.conv2d({
      filters: 64,
      kernelSize: 3,
      activation: "relu",
      padding: "same",
      kernelInitializer: "heNormal",
      name: "conv2a",
    })
  );
  model.add(
    tf.layers.batchNormalization({ name: "bn2a" })
  );
  model.add(
    tf.layers.conv2d({
      filters: 64,
      kernelSize: 3,
      activation: "relu",
      padding: "same",
      kernelInitializer: "heNormal",
      name: "conv2b",
    })
  );
  model.add(
    tf.layers.maxPooling2d({
      poolSize: 2,
      strides: 2,
      name: "pool2",
    })
  );
  model.add(
    tf.layers.dropout({
      rate: 0.25,
      name: "dropout2",
    })
  );

  // ── Conv Block 3 ──
  model.add(
    tf.layers.conv2d({
      filters: 128,
      kernelSize: 3,
      activation: "relu",
      padding: "same",
      kernelInitializer: "heNormal",
      name: "conv3a",
    })
  );
  model.add(
    tf.layers.batchNormalization({ name: "bn3a" })
  );
  model.add(
    tf.layers.conv2d({
      filters: 128,
      kernelSize: 3,
      activation: "relu",
      padding: "same",
      kernelInitializer: "heNormal",
      name: "conv3b",
    })
  );
  model.add(
    tf.layers.maxPooling2d({
      poolSize: 2,
      strides: 2,
      name: "pool3",
    })
  );
  model.add(
    tf.layers.dropout({
      rate: 0.25,
      name: "dropout3",
    })
  );

  // ── Dense Head ──
  model.add(tf.layers.flatten({ name: "flatten" }));

  model.add(
    tf.layers.dense({
      units: 256,
      activation: "relu",
      kernelInitializer: "heNormal",
      name: "dense1",
    })
  );
  model.add(
    tf.layers.batchNormalization({ name: "bn_dense1" })
  );
  model.add(
    tf.layers.dropout({
      rate: 0.5,
      name: "dropout_dense1",
    })
  );

  model.add(
    tf.layers.dense({
      units: 128,
      activation: "relu",
      kernelInitializer: "heNormal",
      name: "dense2",
    })
  );
  model.add(
    tf.layers.dropout({
      rate: 0.3,
      name: "dropout_dense2",
    })
  );

  // ── Output ──
  model.add(
    tf.layers.dense({
      units: MODEL_CONFIG.model.outputClasses,
      activation: "softmax",
      name: "output",
    })
  );

  // Compile with Adam optimizer
  // Using higher learning rate for CNN (0.001 vs 0.0005 for MLP)
  model.compile({
    optimizer: tf.train.adam(MODEL_CONFIG.training.learningRate),
    loss: "categoricalCrossentropy",
    metrics: ["accuracy"],
  });

  return model;
}

/**
 * Print model summary to console.
 * @param {tf.Sequential} model
 */
function printModelSummary(model) {
  console.log("\n=== SignLex CNN Gesture Classifier ===");
  model.summary();
  console.log(`\nTotal params: ${model.countParams()}`);
}

module.exports = { buildGestureModel, printModelSummary };