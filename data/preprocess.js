const fs = require("fs");
const path = require("path");
const sharp = require("sharp");
const { MODEL_CONFIG } = require("../config/modelConfig");
const DatasetLoader = require("./datasetLoader");

const splitConfig = MODEL_CONFIG.splits;

// Label mapping: A=0, B=1, ..., Z=25
const LETTER_TO_LABEL = {};
const LABEL_TO_LETTER = {};
for (let i = 0; i < 26; i++) {
  const letter = String.fromCharCode(65 + i); // A-Z
  LETTER_TO_LABEL[letter] = i;
  LABEL_TO_LETTER[i] = letter;
}

/**
 * Shuffle array in-place using Fisher-Yates algorithm.
 */
function shuffleArray(array) {
  for (let i = array.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [array[i], array[j]] = [array[j], array[i]];
  }
  return array;
}

/**
 * Split dataset into train/validation/test sets.
 */
function splitDataset(data, ratios = splitConfig) {
  const total = ratios.train + ratios.validation + ratios.test;
  if (Math.abs(total - 1.0) > 0.001) {
    throw new Error(`Split ratios must sum to 1.0, got ${total}`);
  }

  const shuffled = shuffleArray([...data]);
  const n = shuffled.length;

  const trainEnd = Math.floor(n * ratios.train);
  const valEnd = trainEnd + Math.floor(n * ratios.validation);

  return {
    train: shuffled.slice(0, trainEnd),
    validation: shuffled.slice(trainEnd, valEnd),
    test: shuffled.slice(valEnd),
  };
}

/**
 * Get class distribution for a dataset.
 */
function getClassDistribution(data) {
  const dist = {};
  for (const sample of data) {
    dist[sample.label] = (dist[sample.label] || 0) + 1;
  }
  return dist;
}

/**
 * Print dataset split statistics.
 */
function printSplitStats(splits) {
  console.log("\n=== Dataset Split Statistics ===");
  console.log(`  Train:      ${splits.train.length} samples`);
  console.log(`  Validation: ${splits.validation.length} samples`);
  console.log(`  Test:       ${splits.test.length} samples`);
  console.log(`  Total:      ${splits.train.length + splits.validation.length + splits.test.length} samples`);

  console.log("\n  Train class distribution:");
  const trainDist = getClassDistribution(splits.train);
  const sortedLabels = Object.keys(trainDist)
    .filter(l => l !== 'undefined')
    .sort((a, b) => parseInt(a) - parseInt(b));
  
  for (const label of sortedLabels) {
    const letter = LABEL_TO_LETTER[label] || '?';
    console.log(`    ${letter} (${label}): ${trainDist[label]} samples`);
  }

  // Check for undefined labels
  if (trainDist['undefined']) {
    console.log(`    WARNING: ${trainDist['undefined']} samples with undefined labels!`);
  }
}

/**
 * Save processed splits to disk using streaming for large files.
 */
async function saveSplits(splits, outputDir) {
  if (!fs.existsSync(outputDir)) {
    fs.mkdirSync(outputDir, { recursive: true });
  }

  for (const [splitName, data] of Object.entries(splits)) {
    const filePath = path.join(outputDir, `${splitName}.json`);
    console.log(`  Writing ${splitName}...`);

    const writeStream = fs.createWriteStream(filePath);
    writeStream.write('[');

    for (let i = 0; i < data.length; i++) {
      const json = JSON.stringify(data[i]);
      writeStream.write(json);
      if (i < data.length - 1) {
        writeStream.write(',');
      }

      if (i % 10000 === 0 && i > 0) {
        console.log(`    Progress: ${i}/${data.length}`);
      }
    }

    writeStream.write(']');

    // Wait for stream to finish
    await new Promise((resolve, reject) => {
      writeStream.on('finish', resolve);
      writeStream.on('error', reject);
      writeStream.end();
    });

    console.log(`  Saved ${splitName}: ${filePath} (${data.length} samples)`);
  }
}

/**
 * Process a single image to 28x28 grayscale, return flattened 784 array.
 */
async function processImage(imagePath) {
  try {
    const buffer = await sharp(imagePath)
      .resize(28, 28, { fit: "fill" })
      .grayscale()
      .raw()
      .toBuffer();

    // Normalize to 0-1
    const pixels = new Array(784);
    for (let i = 0; i < 784; i++) {
      pixels[i] = buffer[i] / 255.0;
    }
    return pixels;
  } catch (err) {
    console.error(`  Failed to process: ${imagePath}`, err.message);
    return null;
  }
}

/**
 * Load ASL Alphabet images and convert to 784-pixel arrays.
 */
async function loadASLAlphabetDataset(aslFiles, maxPerClass = null) {
  const samples = [];
  const validLetters = Object.keys(LETTER_TO_LABEL);

  console.log("\n--- Processing ASL Alphabet Images ---");

  for (const [folderName, filePaths] of Object.entries(aslFiles)) {
    // Only process A-Z folders
    const letter = folderName.toUpperCase();
    if (!validLetters.includes(letter)) {
      console.log(`  Skipping folder: ${folderName} (not A-Z)`);
      continue;
    }

    const label = LETTER_TO_LABEL[letter];
    const filesToProcess = maxPerClass ? filePaths.slice(0, maxPerClass) : filePaths;

    console.log(`  Processing ${letter}: ${filesToProcess.length} images...`);

    let processed = 0;
    let failed = 0;

    for (const filePath of filesToProcess) {
      const pixels = await processImage(filePath);
      if (pixels) {
        samples.push({ pixels, label });
        processed++;
      } else {
        failed++;
      }

      // Progress indicator every 500 images
      if ((processed + failed) % 500 === 0) {
        process.stdout.write(`    ${processed + failed}/${filesToProcess.length}\r`);
      }
    }

    console.log(`    Done: ${processed} processed, ${failed} failed`);
  }

  console.log(`\nTotal ASL Alphabet samples: ${samples.length}`);
  return samples;
}

/**
 * Main preprocessing pipeline.
 */
async function runPreprocessing() {
  console.log("=== SignLex Data Preprocessing Pipeline ===\n");
  console.log("Split ratios:", splitConfig);

  const loader = new DatasetLoader();
  const datasets = await loader.loadAllDatasets();

  const allSamples = [];

  // ── Load Sign MNIST ──
  // Note: Sign MNIST labels are 0-24, skipping 9 (J) - they already match ASL labels!
  // Labels: 0-8 = A-I, 10-24 = K-Y (no J=9, no Z=25)
  if (datasets.signMNIST && datasets.signMNIST.train) {
    console.log("\n--- Loading Sign Language MNIST ---");
    const { images, labels } = datasets.signMNIST.train;

    for (let i = 0; i < labels.length; i++) {
      // Use labels directly - they already match ASL labels (0-24, skipping 9)
      allSamples.push({
        pixels: Array.from(images[i]),
        label: labels[i],
      });
    }
    console.log(`Loaded ${images.length} samples from Sign MNIST`);
    console.log(`  Labels in MNIST: 0-8 (A-I), 10-24 (K-Y) - no J(9) or Z(25)`);

    // Also load test set if available
    if (datasets.signMNIST.test) {
      const { images: testImages, labels: testLabels } = datasets.signMNIST.test;
      for (let i = 0; i < testLabels.length; i++) {
        allSamples.push({
          pixels: Array.from(testImages[i]),
          label: testLabels[i],
        });
      }
      console.log(`Loaded ${testImages.length} samples from Sign MNIST test set`);
    }
  }

  // ── Load ASL Alphabet ──
  // ASL Alphabet has all 26 letters (A-Z), including J and Z
  if (datasets.aslAlphabet) {
    // Set maxPerClass to limit samples per letter (null = all)
    // ASL Alphabet has ~3000 images per letter, total ~78k
    const maxPerClass = null; // Use all images

    const aslSamples = await loadASLAlphabetDataset(datasets.aslAlphabet, maxPerClass);

    for (const sample of aslSamples) {
      allSamples.push({
        pixels: sample.pixels,
        label: sample.label,
      });
    }
  }

  console.log(`\n=== Total Combined Samples: ${allSamples.length} ===`);

  // Verify no undefined labels
  const undefinedCount = allSamples.filter(s => s.label === undefined).length;
  if (undefinedCount > 0) {
    console.error(`ERROR: ${undefinedCount} samples have undefined labels!`);
    process.exit(1);
  }

  // ── Split the data ──
  const splits = splitDataset(allSamples);
  printSplitStats(splits);

  // ── Save to disk ──
  // Rename 'pixels' to 'landmarks' for compatibility with existing training code
  const formatSplit = (data) => data.map((d) => ({ landmarks: d.pixels, label: d.label }));

  const outputDir = MODEL_CONFIG.datasets.aslAlphabet.processedPath;
  console.log(`\nSaving splits to: ${outputDir}`);

  await saveSplits(
    {
      train: formatSplit(splits.train),
      validation: formatSplit(splits.validation),
      test: formatSplit(splits.test),
    },
    outputDir
  );

  console.log("\n=== Preprocessing Complete ===");
}

// Execute if run directly
if (require.main === module) {
  runPreprocessing().catch(console.error);
}

module.exports = {
  shuffleArray,
  splitDataset,
  processImage,
  loadASLAlphabetDataset,
  runPreprocessing,
  LETTER_TO_LABEL,
  LABEL_TO_LETTER,
};