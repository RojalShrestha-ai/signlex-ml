class WebcamCapture {
  constructor(options = {}) {
    this.videoElement = options.videoElement || null;
    this.stream = null;
    this.isRunning = false;
    this.frameCallback = options.onFrame || null;
    this.animationFrameId = null;

    // Default webcam constraints
    this.constraints = {
      video: {
        width: { ideal: 640 },
        height: { ideal: 480 },
        facingMode: "user", 
        frameRate: { ideal: 30 },
      },
      audio: false,
    };

    // Override with custom constraints
    if (options.width) this.constraints.video.width.ideal = options.width;
    if (options.height) this.constraints.video.height.ideal = options.height;
    if (options.frameRate) this.constraints.video.frameRate.ideal = options.frameRate;
  }


  async start() {
    if (this.isRunning) {
      console.warn("Webcam is already running");
      return this.stream;
    }

    try {
      console.log("Requesting webcam access...");
      this.stream = await navigator.mediaDevices.getUserMedia(this.constraints);

      if (this.videoElement) {
        this.videoElement.srcObject = this.stream;
        await this.videoElement.play();
      }

      this.isRunning = true;
      console.log("Webcam started successfully");

      // Log actual stream settings
      const track = this.stream.getVideoTracks()[0];
      const settings = track.getSettings();
      console.log(`  Resolution: ${settings.width}x${settings.height}`);
      console.log(`  Frame rate: ${settings.frameRate}fps`);
      console.log(`  Device: ${track.label}`);

      return this.stream;
    } catch (error) {
      console.error("Failed to access webcam:", error.message);

      // TODO: Provide user-friendly error messages
      if (error.name === "NotAllowedError") {
        throw new Error("Camera permission denied. Please allow camera access.");
      } else if (error.name === "NotFoundError") {
        throw new Error("No camera found. Please connect a webcam.");
      } else {
        throw error;
      }
    }
  }

  /**
   * Stop the webcam stream and release resources.
   */
  stop() {
    if (this.animationFrameId) {
      cancelAnimationFrame(this.animationFrameId);
      this.animationFrameId = null;
    }

    if (this.stream) {
      this.stream.getTracks().forEach((track) => track.stop());
      this.stream = null;
    }

    if (this.videoElement) {
      this.videoElement.srcObject = null;
    }

    this.isRunning = false;
    console.log("Webcam stopped");
  }


  startFrameLoop() {
    if (!this.isRunning) {
      console.error("Cannot start frame loop - webcam not running");
      return;
    }

    if (!this.frameCallback) {
      console.warn("No frame callback registered");
      return;
    }

    // TODO: Implement frame loop
    console.log("Frame loop started (placeholder)");
  }

  /**
   * Check if the browser supports getUserMedia.
   * @returns {boolean}
   */
  static isSupported() {
    return !!(
      navigator.mediaDevices && navigator.mediaDevices.getUserMedia
    );
  }

  /**
   * List available video input devices.
   * @returns {Promise<MediaDeviceInfo[]>}
   */
  static async listCameras() {
    const devices = await navigator.mediaDevices.enumerateDevices();
    return devices.filter((d) => d.kind === "videoinput");
  }
}

// Export for use in the application
if (typeof module !== "undefined" && module.exports) {
  module.exports = WebcamCapture;
}
