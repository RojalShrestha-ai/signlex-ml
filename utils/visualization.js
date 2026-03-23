

const { HAND_CONNECTIONS } = require("./handLandmarks");

// Color scheme for each finger group
const FINGER_COLORS = {
  thumb: "#FF6B6B",       // red
  index: "#4ECDC4",       // teal
  middle: "#45B7D1",      // blue
  ring: "#FFA07A",        // salmon
  pinky: "#98D8C8",       // mint
  palm: "#CCCCCC",        // gray
};

/**
 * Draw hand landmarks and skeleton on a canvas.
 * 
 * @param {CanvasRenderingContext2D} ctx - canvas 2D context
 * @param {number[][]} landmarks - 21 landmarks with [x, y, z]
 * @param {number} width - canvas width
 * @param {number} height - canvas height
 * 
 * TODO: Implement actual canvas drawing
 * TODO: Add landmark point size based on z-depth
 * TODO: Add finger group color coding
 */
function drawHandSkeleton(ctx, landmarks, width, height) {
  if (!landmarks || landmarks.length !== 21) return;

  // TODO: Draw connections
  // for (const [start, end] of HAND_CONNECTIONS) { ... }

  // TODO: Draw landmark points
  // for (let i = 0; i < landmarks.length; i++) { ... }

  console.log("drawHandSkeleton called (not yet implemented)");
}

function clearOverlay(ctx, width, height) {
  ctx.clearRect(0, 0, width, height);
}

module.exports = { drawHandSkeleton, clearOverlay, FINGER_COLORS };
