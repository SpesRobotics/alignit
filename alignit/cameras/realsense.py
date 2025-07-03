import numpy as np
import time
import pyrealsense2 as rs


class RealSense:
    def __init__(self, width, height, fps):
        self.width = width
        self.height = height
        self.pipeline = None
        self.config = None
        self.fps = fps
        print(f"RealSense instance created with a resolution of {width}x{height}.")

    def connect(self):
        print("Initializing Intel RealSense Camera...")

        # Assign to the instance attributes using 'self'
        self.pipeline = rs.pipeline()
        self.config = rs.config()

        # Use the instance attributes to configure and start the stream
        self.config.enable_stream(
            rs.stream.color, self.width, self.height, rs.format.bgr8, self.fps
        )

        try:
            self.pipeline.start(self.config)
            print("RealSense Camera Initialized and pipeline started.")
            # Give camera time to auto-adjust
            time.sleep(2)
        except RuntimeError as e:
            print(f"Error starting RealSense pipeline: {e}")
            print(
                "This usually means the requested resolution is unsupported or the camera is not connected."
            )
            self.pipeline = None  # Reset on failure

    def capture(self):
        frames = self.pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame:
            print("Warning: Could not get color frame. Skipping this pose.")
            return
        image = np.asanyarray(color_frame.get_data())
        print("Captured camera image.")
        return image
