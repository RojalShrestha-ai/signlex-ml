const express = require('express');
const sharp = require('sharp');
const tf = require('@tensorflow/tfjs-node');
const cors = require('cors');
const path = require('path');

const app = express();
const PORT = 3001;

// Middleware
app.use(cors());
app.use(express.json({ limit: '10mb' }));
app.use(express.static(path.join(__dirname)));

// ASL Labels
const LABELS = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'.split('');

let model = null;

/**
 * Load the trained model
 */
async function loadModel() {
  console.log('Loading model...');
  const modelPath = `file://${path.join(__dirname, 'models/saved/gesture_model/model.json')}`;
  model = await tf.loadLayersModel(modelPath);
  console.log('Model loaded successfully!');
  
  // Warm up the model
  const dummy = tf.zeros([1, 784]);
  model.predict(dummy).dispose();
  dummy.dispose();
  console.log('Model warmed up and ready.');
}

/**
 * Process image using Sharp - matching training preprocessing exactly
 */
async function preprocessImage(base64Data) {
  // Remove data URL prefix if present
  const base64Clean = base64Data.replace(/^data:image\/\w+;base64,/, '');
  const imageBuffer = Buffer.from(base64Clean, 'base64');

  // Process with Sharp - EXACTLY like training
  const processed = await sharp(imageBuffer)
    .resize(28, 28, { 
      fit: 'fill',
      kernel: 'lanczos3'  // High quality downsampling
    })
    .grayscale()
    .normalize()  // Auto contrast stretch - key for matching training!
    .raw()
    .toBuffer();

  // Convert to normalized float array (0-1)
  const pixels = new Float32Array(784);
  for (let i = 0; i < 784; i++) {
    pixels[i] = processed[i] / 255.0;
  }

  return pixels;
}

/**
 * Process with additional enhancements for webcam input
 */
async function preprocessImageEnhanced(base64Data) {
  const base64Clean = base64Data.replace(/^data:image\/\w+;base64,/, '');
  const imageBuffer = Buffer.from(base64Clean, 'base64');

  // Enhanced processing for webcam images
  const processed = await sharp(imageBuffer)
    // First, enhance the image
    .modulate({
      brightness: 1.1,      // Slightly brighter
      saturation: 0,        // Ensure grayscale
    })
    .sharpen({
      sigma: 1.5,           // Sharpen to bring out hand details
      m1: 1.0,
      m2: 0.5
    })
    .resize(28, 28, { 
      fit: 'fill',
      kernel: 'lanczos3'
    })
    .grayscale()
    .normalize()            // Stretch contrast to full range
    .gamma(1.2)             // Adjust gamma for better contrast
    .raw()
    .toBuffer();

  const pixels = new Float32Array(784);
  for (let i = 0; i < 784; i++) {
    pixels[i] = processed[i] / 255.0;
  }

  return pixels;
}

/**
 * Run model inference
 */
async function predict(pixels) {
  const inputTensor = tf.tensor2d([Array.from(pixels)], [1, 784]);
  const prediction = model.predict(inputTensor);
  const probabilities = await prediction.data();
  
  inputTensor.dispose();
  prediction.dispose();

  // Find top predictions
  const results = [];
  for (let i = 0; i < probabilities.length; i++) {
    results.push({ letter: LABELS[i], confidence: probabilities[i] });
  }
  results.sort((a, b) => b.confidence - a.confidence);

  return {
    prediction: results[0].letter,
    confidence: results[0].confidence,
    top3: results.slice(0, 3)
  };
}

/**
 * Debug endpoint - returns the processed image so you can see what model sees
 */
app.post('/api/debug-preprocess', async (req, res) => {
  try {
    const { image } = req.body;
    if (!image) {
      return res.status(400).json({ error: 'No image provided' });
    }

    const base64Clean = image.replace(/^data:image\/\w+;base64,/, '');
    const imageBuffer = Buffer.from(base64Clean, 'base64');

    // Process and return as PNG for visualization
    const processed = await sharp(imageBuffer)
      .resize(28, 28, { fit: 'fill', kernel: 'lanczos3' })
      .grayscale()
      .normalize()
      .png()
      .toBuffer();

    const processedBase64 = `data:image/png;base64,${processed.toString('base64')}`;
    
    res.json({ processedImage: processedBase64 });
  } catch (err) {
    console.error('Debug preprocess error:', err);
    res.status(500).json({ error: err.message });
  }
});

/**
 * Main prediction endpoint
 */
app.post('/api/predict', async (req, res) => {
  try {
    const { image, enhanced = true } = req.body;
    
    if (!image) {
      return res.status(400).json({ error: 'No image provided' });
    }

    if (!model) {
      return res.status(503).json({ error: 'Model not loaded yet' });
    }

    // Preprocess the image
    const pixels = enhanced 
      ? await preprocessImageEnhanced(image)
      : await preprocessImage(image);

    // Run inference
    const result = await predict(pixels);

    res.json(result);
  } catch (err) {
    console.error('Prediction error:', err);
    res.status(500).json({ error: err.message });
  }
});

/**
 * Health check endpoint
 */
app.get('/api/health', (req, res) => {
  res.json({ 
    status: 'ok', 
    modelLoaded: model !== null,
    timestamp: new Date().toISOString()
  });
});

// Start server
async function start() {
  await loadModel();
  
  app.listen(PORT, () => {
    console.log(`\n🚀 SignLex server running at http://localhost:${PORT}`);
    console.log(`   - Web UI: http://localhost:${PORT}/index.html`);
    console.log(`   - API: http://localhost:${PORT}/api/predict`);
    console.log(`   - Health: http://localhost:${PORT}/api/health`);
  });
}

start().catch(console.error);