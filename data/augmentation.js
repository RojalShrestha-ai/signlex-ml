const { MODEL_CONFIG } = require("../config/modelConfig");

const augConfig = MODEL_CONFIG.augmentation;

/**
 * Apply random rotation to landmark coordinates.
 * Rotates all 21 landmarks around the hand center by a random angle.
 * 
 * @param {number[][]} landmarks - array of [x, y, z] coordinates
 * @param {number} maxDegrees - maximum rotation in degrees
 * @returns {number[][]} rotated landmarks
 */
function randomRotation(landmarks, maxDegrees = augConfig.rotationRange) {
  const angle = (Math.random() * 2 - 1) * maxDegrees * (Math.PI / 180);
  const cosA = Math.cos(angle);
  const sinA = Math.sin(angle);

  // Calculate centroid of all landmarks
  let cx = 0, cy = 0;
  for (const [x, y] of landmarks) {
    cx += x;
    cy += y;
  }
  cx /= landmarks.length;
  cy /= landmarks.length;

  // Rotate each landmark around the centroid
  return landmarks.map(([x, y, z]) => {
    const dx = x - cx;
    const dy = y - cy;
    return [
      cx + dx * cosA - dy * sinA,
      cy + dx * sinA + dy * cosA,
      z, // z-coordinate unchanged for 2D rotation
    ];
  });
}

/**
 * Apply random brightness adjustment to pixel data.
 * Multiplies pixel values by a random factor within the configured range.
 * 
 * @param {Float32Array} pixels - normalized pixel values [0, 1]
 * @param {number} range - brightness variation factor
 * @returns {Float32Array} adjusted pixels, clamped to [0, 1]
 */
function randomBrightness(pixels, range = augConfig.brightnessRange) {
  const factor = 1.0 + (Math.random() * 2 - 1) * range;
  const result = new Float32Array(pixels.length);

  for (let i = 0; i < pixels.length; i++) {
    result[i] = Math.max(0, Math.min(1, pixels[i] * factor));
  }

  return result;
}

/**
 * Horizontally flip landmark coordinates.
 * Mirrors the x-coordinate around 0.5 (center of normalized space).
 * 
 * @param {number[][]} landmarks - array of [x, y, z] coordinates
 * @returns {number[][]} flipped landmarks
 */
function horizontalFlip(landmarks) {
  if (!augConfig.horizontalFlip || Math.random() > 0.5) {
    return landmarks; // 50% chance to skip flip
  }

  return landmarks.map(([x, y, z]) => [1.0 - x, y, z]);
}

/**
 * Add random noise to landmark coordinates to simulate detection jitter.
 * 
 * @param {number[][]} landmarks - array of [x, y, z] coordinates
 * @param {number} noiseScale - standard deviation of Gaussian noise
 * @returns {number[][]} noisy landmarks
 */
function addLandmarkNoise(landmarks, noiseScale = 0.02) {
  return landmarks.map(([x, y, z]) => [
    x + gaussianRandom() * noiseScale,
    y + gaussianRandom() * noiseScale,
    z + gaussianRandom() * noiseScale * 0.5, // less noise on z-axis
  ]);
}

/**
 * Generate a random number from a Gaussian distribution (Box-Muller).
 * @returns {number} random value from N(0, 1)
 */
function gaussianRandom() {
  let u = 0, v = 0;
  while (u === 0) u = Math.random();
  while (v === 0) v = Math.random();
  return Math.sqrt(-2.0 * Math.log(u)) * Math.cos(2.0 * Math.PI * v);
}

/**
 * Apply random scaling (zoom) to landmark coordinates.
 * 
 * @param {number[][]} landmarks - array of [x, y, z] coordinates
 * @param {number} zoomRange - max zoom factor deviation from 1.0
 * @returns {number[][]} zoomed landmarks
 */
function randomZoom(landmarks, zoomRange = augConfig.zoomRange) {
  const scale = 1.0 + (Math.random() * 2 - 1) * zoomRange;

  // Calculate centroid
  let cx = 0, cy = 0;
  for (const [x, y] of landmarks) {
    cx += x;
    cy += y;
  }
  cx /= landmarks.length;
  cy /= landmarks.length;

  // Scale around centroid
  return landmarks.map(([x, y, z]) => [
    cx + (x - cx) * scale,
    cy + (y - cy) * scale,
    z,
  ]);
}

/**
 * Compose multiple augmentation transforms into a single pipeline.
 * Randomly applies a subset of transforms to each sample.
 * 
 * @param {number[][]} landmarks - original landmark data
 * @param {Object} options - which transforms to enable
 * @returns {number[][]} augmented landmark data
 */
function augmentLandmarks(landmarks, options = {}) {
  let result = [...landmarks.map((l) => [...l])]; // deep copy

  const {
    rotate = true,
    flip = true,
    zoom = true,
    noise = true,
  } = options;

  if (rotate && Math.random() > 0.3) {
    result = randomRotation(result);
  }

  if (flip) {
    result = horizontalFlip(result);
  }

  if (zoom && Math.random() > 0.3) {
    result = randomZoom(result);
  }

  if (noise && Math.random() > 0.5) {
    result = addLandmarkNoise(result);
  }

  return result;
}

/**
 * Generate N augmented copies of a single landmark sample.
 * 
 * @param {number[][]} landmarks - original landmarks
 * @param {number} label - class label
 * @param {number} numCopies - how many augmented samples to create
 * @returns {Object[]} array of { landmarks, label } pairs
 */
function generateAugmentedSamples(landmarks, label, numCopies = 5) {
  const samples = [{ landmarks, label }]; // include original

  for (let i = 0; i < numCopies; i++) {
    samples.push({
      landmarks: augmentLandmarks(landmarks),
      label,
    });
  }

  return samples;
}

module.exports = {
  randomRotation,
  randomBrightness,
  horizontalFlip,
  addLandmarkNoise,
  randomZoom,
  augmentLandmarks,
  generateAugmentedSamples,
};
