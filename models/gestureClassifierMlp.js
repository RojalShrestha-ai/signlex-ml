
const { MODEL_CONFIG } = require("../config/modelConfig");

/**
 * Build the gesture classification model.
 * Uses TensorFlow.js sequential API with dense layers.
 * 
 * @param {Object} tf - TensorFlow.js module (injected for flexibility)
 * @returns {tf.Sequential} compiled model ready for training
 */
function buildGestureModel(tf) {
  const config = MODEL_CONFIG.model;
  const trainConfig = MODEL_CONFIG.training;

  const model = tf.sequential({ name: "signlex-gesture-classifier" });

  // Input layer + first hidden layer
  model.add(
    tf.layers.dense({
      inputShape: config.inputShape,
      units: config.hiddenLayers[0],
      activation: config.activation,
      kernelInitializer: "heNormal",
      name: "hidden_1",
    })
  );

  // Dropout for regularization
  model.add(
    tf.layers.dropout({
      rate: config.dropout,
      name: "dropout_1",
    })
  );

  // Second hidden layer
  model.add(
    tf.layers.dense({
      units: config.hiddenLayers[1],
      activation: config.activation,
      kernelInitializer: "heNormal",
      name: "hidden_2",
    })
  );

  // Dropout
  model.add(
    tf.layers.dropout({
      rate: config.dropout,
      name: "dropout_2",
    })
  );

  model.add(tf.layers.dense({ units: 64, activation: config.activation, kernelInitializer: 'heNormal', name: 'hidden_3' }));
  model.add(tf.layers.dropout({ rate: config.dropout, name: 'dropout_3' }));

  // Output layer
  model.add(
    tf.layers.dense({
      units: config.outputClasses,
      activation: config.outputActivation,
      name: "output",
    })
  );

  // Compile the model
  model.compile({
    optimizer: tf.train.adam(trainConfig.learningRate),
    loss: trainConfig.lossFunction,
    metrics: trainConfig.metrics,
  });

  return model;
}

/**
 * Print model summary to console.
 * @param {tf.Sequential} model
 */
function printModelSummary(model) {
  console.log("\n=== SignLex Gesture Classifier Model ===");
  model.summary();
  console.log(`\nTotal params: ${model.countParams()}`);
}

// TODO: Implement training pipeline
// async function trainModel(model, trainData, valData, config) { ... }

// TODO: Implement model evaluation
// async function evaluateModel(model, testData) { ... }

// TODO: Implement TF.js model export
// async function exportToTFJS(model, outputPath) { ... }

module.exports = { buildGestureModel, printModelSummary };
