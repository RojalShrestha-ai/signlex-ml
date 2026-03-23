const fs = require("fs");
const path = require("path");
const JSONStream = require("JSONStream");

// Use GPU version of TensorFlow.js
const tf = require("@tensorflow/tfjs-node-gpu");

const { MODEL_CONFIG } = require("../config/modelConfig");
const { buildGestureModel, printModelSummary } = require("./gestureClassifier");

/**
 * Stream-read a large JSON array file.
 */
function loadLargeJSON(filePath) {
  return new Promise((resolve, reject) => {
    const data = [];
    const stream = fs.createReadStream(filePath, { encoding: "utf8" });
    const parser = JSONStream.parse("*");

    parser.on("data", (obj) => {
      data.push(obj);
      if (data.length % 10000 === 0) {
        process.stdout.write(`    Loaded ${data.length} samples...\r`);
      }
    });

    parser.on("end", () => {
      console.log(`    Loaded ${data.length} samples`);
      resolve(data);
    });

    parser.on("error", (err) => reject(err));
    stream.pipe(parser);
  });
}

async function main() {
  console.log("=== SignLex Model Training Pipeline (GPU) ===\n");
  console.log(`TensorFlow.js version: ${tf.version.tfjs}`);
  console.log(`Backend: ${tf.getBackend()}`);
  console.log("");

  // ── Load preprocessed data ──
  const trainDataPath = path.join(MODEL_CONFIG.datasets.aslAlphabet.processedPath, "train.json");
  const valDataPath = path.join(MODEL_CONFIG.datasets.aslAlphabet.processedPath, "validation.json");

  if (!fs.existsSync(trainDataPath)) {
    console.error("train.json not found. Run `npm run preprocess` first.");
    process.exit(1);
  }

  console.log("Loading preprocessed data...");
  console.log("  Loading train.json...");
  const trainData = await loadLargeJSON(trainDataPath);
  console.log("  Loading validation.json...");
  const valData = await loadLargeJSON(valDataPath);
  console.log(`\n  Train samples:      ${trainData.length}`);
  console.log(`  Validation samples: ${valData.length}\n`);

  // ── Build model ──
  const model = buildGestureModel(tf);
  printModelSummary(model);

  // ── Convert data to tensors ──
  console.log("\nConverting data to tensors...");

  const numClasses = MODEL_CONFIG.model.outputClasses;

  // Create X tensors
  const trainX = tf.tensor2d(trainData.map((d) => d.landmarks));
  const valX = tf.tensor2d(valData.map((d) => d.landmarks));

  // Manual one-hot encoding (avoids buggy tf.oneHot in tfjs-node-gpu)
  console.log("  Creating one-hot encoded labels...");
  const trainYData = trainData.map((d) => {
    const arr = new Array(numClasses).fill(0);
    arr[d.label] = 1;
    return arr;
  });
  const valYData = valData.map((d) => {
    const arr = new Array(numClasses).fill(0);
    arr[d.label] = 1;
    return arr;
  });

  const trainY = tf.tensor2d(trainYData);
  const valY = tf.tensor2d(valYData);

  console.log(`  trainX shape: [${trainX.shape}]`);
  console.log(`  trainY shape: [${trainY.shape}]`);
  console.log(`  valX shape:   [${valX.shape}]`);
  console.log(`  valY shape:   [${valY.shape}]`);

  // ── Create output dirs ──
  const { modelSavePath, logsPath } = MODEL_CONFIG.output;
  [modelSavePath, logsPath].forEach((p) => {
    if (!fs.existsSync(p)) fs.mkdirSync(p, { recursive: true });
  });

  // ── Train ──
  console.log("\n=== Starting Training ===");
  console.log(`  Epochs:        ${MODEL_CONFIG.training.epochs}`);
  console.log(`  Batch size:    ${MODEL_CONFIG.training.batchSize}`);
  console.log(`  Learning rate: ${MODEL_CONFIG.training.learningRate}`);
  console.log(`  Early stopping patience: ${MODEL_CONFIG.training.earlyStopping.patience}`);
  console.log("");

  let bestValLoss = Infinity;
  let bestValAcc = 0;
  let patienceCounter = 0;
  const { patience, minDelta } = MODEL_CONFIG.training.earlyStopping;

  const startTime = Date.now();

  await model.fit(trainX, trainY, {
    epochs: MODEL_CONFIG.training.epochs,
    batchSize: MODEL_CONFIG.training.batchSize,
    validationData: [valX, valY],
    shuffle: true,
    callbacks: {
      onEpochEnd: async (epoch, logs) => {
        const acc = logs.acc ?? logs.accuracy;
        const valAcc = logs.val_acc ?? logs.val_accuracy;

        console.log(
          `Epoch ${String(epoch + 1).padStart(2, "0")}/${MODEL_CONFIG.training.epochs}` +
            `  loss: ${logs.loss.toFixed(4)}` +
            `  acc: ${acc.toFixed(4)}` +
            `  val_loss: ${logs.val_loss.toFixed(4)}` +
            `  val_acc: ${valAcc.toFixed(4)}`
        );

        if (valAcc > bestValAcc) {
          bestValAcc = valAcc;
        }

        if (logs.val_loss < bestValLoss - minDelta) {
          bestValLoss = logs.val_loss;
          patienceCounter = 0;

          await model.save(
            tf.io.withSaveHandler(async (artifacts) => {
              fs.writeFileSync(
                path.join(modelSavePath, "model.json"),
                JSON.stringify({
                  modelTopology: artifacts.modelTopology,
                  weightsManifest: [
                    {
                      paths: ["weights.bin"],
                      weights: artifacts.weightSpecs,
                    },
                  ],
                })
              );
              fs.writeFileSync(path.join(modelSavePath, "weights.bin"), Buffer.from(artifacts.weightData));
              return { modelArtifactsInfo: { dateSaved: new Date(), modelTopologyType: "JSON" } };
            })
          );
          console.log(`    >> Saved best model (val_loss: ${bestValLoss.toFixed(4)}, val_acc: ${valAcc.toFixed(4)})`);
        } else {
          patienceCounter++;
          if (patienceCounter >= patience) {
            console.log(`\nEarly stopping triggered after ${epoch + 1} epochs.`);
            model.stopTraining = true;
          }
        }
      },
    },
  });

  const trainingTime = ((Date.now() - startTime) / 1000).toFixed(1);

  console.log("\n=== Training Complete ===");
  console.log(`  Training time:  ${trainingTime}s`);
  console.log(`  Best val_loss:  ${bestValLoss.toFixed(4)}`);
  console.log(`  Best val_acc:   ${(bestValAcc * 100).toFixed(2)}%`);
  console.log(`  Model saved to: ${modelSavePath}`);

  console.log("\n=== Final Evaluation ===");
  const evalResult = model.evaluate(valX, valY);
  const [finalLoss, finalAcc] = await Promise.all([evalResult[0].data(), evalResult[1].data()]);
  console.log(`  Val Loss:     ${finalLoss[0].toFixed(4)}`);
  console.log(`  Val Accuracy: ${(finalAcc[0] * 100).toFixed(2)}%`);

  trainX.dispose();
  trainY.dispose();
  valX.dispose();
  valY.dispose();
  evalResult.forEach((t) => t.dispose());

  console.log("\nDone!");
}

main().catch(console.error);