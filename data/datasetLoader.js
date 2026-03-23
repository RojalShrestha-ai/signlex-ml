const fs = require("fs");
const path = require("path");
const csv = require("csv-parser");
const { MODEL_CONFIG } = require("../config/modelConfig");

class DatasetLoader {
  constructor() {
    this.config = MODEL_CONFIG.datasets;
    this.loadedData = {
      aslAlphabet: { images: [], labels: [] },
      signMNIST: { train: null, test: null },
    };
  }

  /**
   * Scan the ASL Alphabet dataset directory structure.
   * Expected format: raw/asl_alphabet/asl_alphabet_train/A/*.jpg
   * Each subdirectory is a class label (A-Z, del, nothing, space)
   * 
   * @returns {Object} mapping of label -> array of file paths
   */
  scanASLAlphabetDirectory() {
    const basePath = this.config.aslAlphabet.rawPath;
    const trainPath = path.join(basePath, "asl_alphabet_train");
    const classDirs = {};

    if (!fs.existsSync(trainPath)) {
      console.error(`Dataset not found at: ${trainPath}`);
      console.log("Please download from:", this.config.aslAlphabet.source);
      console.log("Extract to:", basePath);
      return null;
    }

    const directories = fs
      .readdirSync(trainPath, { withFileTypes: true })
      .filter((dirent) => dirent.isDirectory())
      .map((dirent) => dirent.name);

    for (const dir of directories) {
      const dirPath = path.join(trainPath, dir);
      const files = fs
        .readdirSync(dirPath)
        .filter((f) => f.endsWith(".jpg") || f.endsWith(".png"));

      classDirs[dir] = files.map((f) => path.join(dirPath, f));
    }

    console.log(`Found ${directories.length} classes in ASL Alphabet dataset`);
    for (const [label, files] of Object.entries(classDirs)) {
      console.log(`  ${label}: ${files.length} images`);
    }

    return classDirs;
  }

  /**
   * Load the Sign Language MNIST dataset from CSV format.
   * CSV format: label, pixel1, pixel2, ..., pixel784
   * Each row is a 28x28 flattened grayscale image.
   * 
   * @param {string} csvPath - path to the CSV file
   * @returns {Promise<Object>} { images: Float32Array[], labels: number[] }
   */
  loadSignMNISTFromCSV(csvPath) {
    return new Promise((resolve, reject) => {
      const images = [];
      const labels = [];

      if (!fs.existsSync(csvPath)) {
        console.error(`MNIST CSV not found at: ${csvPath}`);
        console.log("Please download from:", this.config.signLanguageMNIST.source);
        reject(new Error("Dataset file not found"));
        return;
      }

      fs.createReadStream(csvPath)
        .pipe(csv())
        .on("data", (row) => {
          const label = parseInt(row.label, 10);
          const pixels = [];

          // Extract 784 pixel values (28x28 image)
          for (let i = 0; i < 784; i++) {
            const pixelKey = `pixel${i + 1}`;
            if (row[pixelKey] !== undefined) {
              // Normalize pixel values to [0, 1]
              pixels.push(parseFloat(row[pixelKey]) / 255.0);
            }
          }

          labels.push(label);
          images.push(new Float32Array(pixels));
        })
        .on("end", () => {
          console.log(`Loaded ${images.length} samples from ${csvPath}`);
          console.log(`Labels range: ${Math.min(...labels)} to ${Math.max(...labels)}`);
          resolve({ images, labels });
        })
        .on("error", (err) => {
          reject(err);
        });
    });
  }

  /**
   * Load both datasets and return combined data structure.
   * 
   * @returns {Promise<Object>} combined dataset object
   */
  async loadAllDatasets() {
    console.log("=== SignLex Dataset Loader ===\n");

    // Scan ASL Alphabet directory
    console.log("1. Scanning ASL Alphabet Dataset...");
    const aslFiles = this.scanASLAlphabetDirectory();

    // Load Sign Language MNIST
    console.log("\n2. Loading Sign Language MNIST...");
    const mnistTrainPath = path.join(
      this.config.signLanguageMNIST.rawPath,
      "sign_mnist_train.csv"
    );
    const mnistTestPath = path.join(
      this.config.signLanguageMNIST.rawPath,
      "sign_mnist_test.csv"
    );

    let mnistTrain = null;
    let mnistTest = null;

    try {
      mnistTrain = await this.loadSignMNISTFromCSV(mnistTrainPath);
      mnistTest = await this.loadSignMNISTFromCSV(mnistTestPath);
    } catch (err) {
      console.warn("MNIST loading skipped:", err.message);
    }

    return {
      aslAlphabet: aslFiles,
      signMNIST: {
        train: mnistTrain,
        test: mnistTest,
      },
    };
  }

  /**
   * Get dataset statistics for reporting.
   * @returns {Object} stats object
   */
  getDatasetStats() {
    // TODO: Implement full statistics after loading
    return {
      aslAlphabet: {
        totalClasses: this.config.aslAlphabet.numClasses,
        imageSize: this.config.aslAlphabet.imageSize,
        estimated_total: 87000,
      },
      signMNIST: {
        totalClasses: this.config.signLanguageMNIST.numClasses,
        imageSize: this.config.signLanguageMNIST.imageSize,
        trainSamples: 27455,
        testSamples: 7172,
      },
    };
  }
}

module.exports = DatasetLoader;
