const { MODEL_CONFIG } = require("../config/modelConfig");

// MediaPipe Hand Landmark indices
const HAND_LANDMARKS = {
  WRIST: 0,
  THUMB_CMC: 1,
  THUMB_MCP: 2,
  THUMB_IP: 3,
  THUMB_TIP: 4,
  INDEX_FINGER_MCP: 5,
  INDEX_FINGER_PIP: 6,
  INDEX_FINGER_DIP: 7,
  INDEX_FINGER_TIP: 8,
  MIDDLE_FINGER_MCP: 9,
  MIDDLE_FINGER_PIP: 10,
  MIDDLE_FINGER_DIP: 11,
  MIDDLE_FINGER_TIP: 12,
  RING_FINGER_MCP: 13,
  RING_FINGER_PIP: 14,
  RING_FINGER_DIP: 15,
  RING_FINGER_TIP: 16,
  PINKY_MCP: 17,
  PINKY_PIP: 18,
  PINKY_DIP: 19,
  PINKY_TIP: 20,
};

// Connection pairs for skeleton overlay visualization
const HAND_CONNECTIONS = [
  // Thumb
  [0, 1], [1, 2], [2, 3], [3, 4],
  // Index finger
  [0, 5], [5, 6], [6, 7], [7, 8],
  // Middle finger
  [0, 9], [9, 10], [10, 11], [11, 12],
  // Ring finger
  [0, 13], [13, 14], [14, 15], [15, 16],
  // Pinky
  [0, 17], [17, 18], [18, 19], [19, 20],
  // Palm connections
  [5, 9], [9, 13], [13, 17],
];

class HandLandmarkExtractor {
  constructor() {
    this.mpConfig = MODEL_CONFIG.mediapipe;
    this.hands = null;
    this.isInitialized = false;
    this.latestResults = null;
    this.onLandmarksCallback = null;
  }

  /**
   * Initialize the MediaPipe Hands solution.
   * Loads the WASM binary and ML model files.
   * 
   * Note: This requires @mediapipe/hands to be installed
   * and runs in a browser environment with WASM support.
   * 
   * TODO: Complete initialization with actual MediaPipe import
   * TODO: Handle WASM loading errors
   * TODO: Add model warm-up for first-frame latency
   */
  async initialize() {
    console.log("Initializing MediaPipe Hands...");
    console.log("  Max hands:", this.mpConfig.maxNumHands);
    console.log("  Detection confidence:", this.mpConfig.minDetectionConfidence);
    console.log("  Tracking confidence:", this.mpConfig.minTrackingConfidence);
    console.log("  Model complexity:", this.mpConfig.modelComplexity);

    // TODO: Import and configure MediaPipe Hands
    // const { Hands } = await import("@mediapipe/hands");
    //
    // this.hands = new Hands({
    //   locateFile: (file) =>
    //     `https://cdn.jsdelivr.net/npm/@mediapipe/hands/${file}`,
    // });
    //
    // this.hands.setOptions({
    //   maxNumHands: this.mpConfig.maxNumHands,
    //   minDetectionConfidence: this.mpConfig.minDetectionConfidence,
    //   minTrackingConfidence: this.mpConfig.minTrackingConfidence,
    //   modelComplexity: this.mpConfig.modelComplexity,
    // });
    //
    // this.hands.onResults(this.onResults.bind(this));

    this.isInitialized = true;
    console.log("MediaPipe Hands initialized (placeholder)");
  }

  /**
   * Process a single video frame through MediaPipe Hands.
   * 
   * @param {HTMLVideoElement|HTMLCanvasElement} frame - input frame
   * @returns {Promise<Object|null>} detection results or null
   * 
   * TODO: Implement actual frame processing
   */
  async processFrame(frame) {
    if (!this.isInitialized) {
      console.error("HandLandmarkExtractor not initialized. Call initialize() first.");
      return null;
    }

    // TODO: Send frame to MediaPipe
    // await this.hands.send({ image: frame });
    // return this.latestResults;

    return null; // placeholder
  }


  onResults(results) {
    this.latestResults = results;

    if (results.multiHandLandmarks && results.multiHandLandmarks.length > 0) {
      const landmarks = results.multiHandLandmarks[0]; // first hand
      const handedness = results.multiHandedness
        ? results.multiHandedness[0]
        : null;

      // Convert MediaPipe landmarks to our format: [[x, y, z], ...]
      const landmarkArray = landmarks.map((lm) => [lm.x, lm.y, lm.z]);

      if (this.onLandmarksCallback) {
        this.onLandmarksCallback({
          landmarks: landmarkArray,
          handedness: handedness,
          timestamp: Date.now(),
        });
      }
    }
  }


  setOnLandmarks(callback) {
    this.onLandmarksCallback = callback;
  }

  static getConnections() {
    return HAND_CONNECTIONS;
  }

  static getLandmarkName(index) {
    const entry = Object.entries(HAND_LANDMARKS).find(
      ([, val]) => val === index
    );
    return entry ? entry[0] : `UNKNOWN_${index}`;
  }

  destroy() {
    if (this.hands) {
      this.hands.close();
      this.hands = null;
    }
    this.isInitialized = false;
    this.latestResults = null;
    console.log("HandLandmarkExtractor destroyed");
  }
}

if (typeof module !== "undefined" && module.exports) {
  module.exports = { HandLandmarkExtractor, HAND_LANDMARKS, HAND_CONNECTIONS };
}
