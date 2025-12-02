#
# RealSense D455 Streaming Module for RTG-SLAM
#

import threading
import queue
import time
from typing import NamedTuple, Optional
import numpy as np
from PIL import Image

try:
    import pyrealsense2 as rs
except ImportError:
    raise ImportError("Please install pyrealsense2: pip install pyrealsense2")

from scene.dataset_readers import CameraInfo
from utils.graphics_utils import focal2fov


class RealsenseConfig(NamedTuple):
    """RealSense camera configuration"""
    width: int = 640
    height: int = 480
    fps: int = 30
    depth_scale: float = 1000.0  # depth in mm -> meters
    align_depth: bool = True
    enable_auto_exposure: bool = True
    enable_auto_white_balance: bool = True
    color_temperature: Optional[int] = None  # Color temperature in Kelvin (3000-6500), None for auto
    exposure: Optional[int] = None  # Manual exposure value, None for auto
    brightness: int = 0  # -64 to 64
    contrast: int = 50  # 0 to 100
    saturation: int = 64  # 0 to 100
    hue: int = 0  # -180 to 180 (for color shift adjustment)


class RealsenseStream:
    """
    Real-time streaming from RealSense D455 camera.
    Provides an iterator interface compatible with RTG-SLAM's main loop.
    """

    def __init__(
        self,
        config: Optional[RealsenseConfig] = None,
        queue_size: int = 2,
        crop_edge: int = 0,
    ):
        self.config = config or RealsenseConfig()
        self.queue_size = queue_size
        self.crop_edge = crop_edge

        # RealSense pipeline
        self.pipeline = None
        self.profile = None
        self.align = None

        # Threading
        self.frame_queue = queue.Queue(maxsize=queue_size)
        self.capture_thread = None
        self.running = False

        # Frame counter
        self.frame_id = 0
        self.start_time = None

        # Intrinsics (will be set after pipeline start)
        self.intrinsics = None
        self.depth_scale = None

    def start(self):
        """Initialize and start the RealSense pipeline"""
        self.pipeline = rs.pipeline()
        config = rs.config()

        # Configure streams
        config.enable_stream(
            rs.stream.color,
            self.config.width,
            self.config.height,
            rs.format.rgb8,
            self.config.fps
        )
        config.enable_stream(
            rs.stream.depth,
            self.config.width,
            self.config.height,
            rs.format.z16,
            self.config.fps
        )

        # Start pipeline
        self.profile = self.pipeline.start(config)

        # Get depth scale
        depth_sensor = self.profile.get_device().first_depth_sensor()
        self.depth_scale = depth_sensor.get_depth_scale()

        # Configure color sensor (white balance, exposure, etc.)
        color_sensor = self.profile.get_device().first_color_sensor()
        
        # Auto exposure
        if self.config.enable_auto_exposure:
            color_sensor.set_option(rs.option.enable_auto_exposure, 1)
        else:
            color_sensor.set_option(rs.option.enable_auto_exposure, 0)
            if self.config.exposure is not None:
                color_sensor.set_option(rs.option.exposure, self.config.exposure)
        
        # Auto white balance
        if self.config.enable_auto_white_balance:
            color_sensor.set_option(rs.option.enable_auto_white_balance, 1)
        else:
            color_sensor.set_option(rs.option.enable_auto_white_balance, 0)
            if self.config.color_temperature is not None:
                color_sensor.set_option(rs.option.color_temperature, self.config.color_temperature)
        
        # Adjust color properties
        color_sensor.set_option(rs.option.brightness, self.config.brightness)
        color_sensor.set_option(rs.option.contrast, self.config.contrast)
        color_sensor.set_option(rs.option.saturation, self.config.saturation)
        color_sensor.set_option(rs.option.hue, self.config.hue)

        # Create align object
        if self.config.align_depth:
            self.align = rs.align(rs.stream.color)

        # Get intrinsics
        color_stream = self.profile.get_stream(rs.stream.color)
        self.intrinsics = color_stream.as_video_stream_profile().get_intrinsics()

        # Start capture thread
        self.running = True
        self.start_time = time.time()
        self.capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.capture_thread.start()

        print(f"[RealsenseStream] Started with resolution {self.config.width}x{self.config.height} @ {self.config.fps}fps")
        print(f"[RealsenseStream] Intrinsics: fx={self.intrinsics.fx:.2f}, fy={self.intrinsics.fy:.2f}, "
              f"cx={self.intrinsics.ppx:.2f}, cy={self.intrinsics.ppy:.2f}")
        print(f"[RealsenseStream] Color settings - Brightness: {self.config.brightness}, Contrast: {self.config.contrast}, "
              f"Saturation: {self.config.saturation}, Hue: {self.config.hue}")

        return self

    def stop(self):
        """Stop the RealSense pipeline"""
        self.running = False
        if self.capture_thread:
            self.capture_thread.join(timeout=2.0)
        if self.pipeline:
            self.pipeline.stop()
        print("[RealsenseStream] Stopped")

    def _capture_loop(self):
        """Background thread for continuous frame capture"""
        while self.running:
            try:
                # Wait for frames with timeout
                frames = self.pipeline.wait_for_frames(timeout_ms=1000)

                # Align depth to color if enabled
                if self.align:
                    frames = self.align.process(frames)

                color_frame = frames.get_color_frame()
                depth_frame = frames.get_depth_frame()

                if not color_frame or not depth_frame:
                    continue

                # Convert to numpy
                color_image = np.asanyarray(color_frame.get_data())
                depth_image = np.asanyarray(depth_frame.get_data()).astype(np.float32)

                # Apply depth scale (convert to meters)
                depth_image = depth_image * self.depth_scale

                timestamp = time.time() - self.start_time

                # Try to put in queue (non-blocking to avoid backlog)
                try:
                    self.frame_queue.put_nowait((color_image, depth_image, timestamp))
                except queue.Full:
                    # Drop oldest frame if queue is full
                    try:
                        self.frame_queue.get_nowait()
                        self.frame_queue.put_nowait((color_image, depth_image, timestamp))
                    except queue.Empty:
                        pass

            except Exception as e:
                if self.running:
                    print(f"[RealsenseStream] Capture error: {e}")

    def get_camera_info(self, color_image: np.ndarray, depth_image: np.ndarray, timestamp: float) -> CameraInfo:
        """Convert raw frames to CameraInfo"""
        fx = self.intrinsics.fx
        fy = self.intrinsics.fy
        cx = self.intrinsics.ppx
        cy = self.intrinsics.ppy
        width = self.intrinsics.width
        height = self.intrinsics.height

        # Apply crop if needed
        if self.crop_edge > 0:
            ce = self.crop_edge
            color_image = color_image[ce:-ce, ce:-ce, :]
            depth_image = depth_image[ce:-ce, ce:-ce]
            cx -= ce
            cy -= ce
            width -= 2 * ce
            height -= 2 * ce

        # Convert to PIL Image
        image_pil = Image.fromarray(color_image)
        depth_pil = Image.fromarray(depth_image)

        # Calculate FoV
        FovX = focal2fov(fx, width)
        FovY = focal2fov(fy, height)

        # Initial pose (identity - will be updated by tracker)
        R = np.eye(3)
        T = np.zeros(3)
        pose_gt = np.eye(4)

        return CameraInfo(
            uid=self.frame_id,
            R=R,
            T=T,
            FovY=FovY,
            FovX=FovX,
            image=image_pil,
            image_path=f"realsense_frame_{self.frame_id:06d}",
            image_name=f"frame_{self.frame_id:06d}",
            width=width,
            height=height,
            depth=depth_pil,
            depth_path=f"realsense_depth_{self.frame_id:06d}",
            pose_gt=pose_gt,
            cx=cx,
            cy=cy,
            depth_scale=1.0,  # Already converted to meters
            timestamp=timestamp,
        )

    def __iter__(self):
        """Make the stream iterable"""
        return self

    def __next__(self) -> CameraInfo:
        """Get next frame as CameraInfo"""
        if not self.running:
            raise StopIteration

        try:
            # Block waiting for next frame
            color_image, depth_image, timestamp = self.frame_queue.get(timeout=5.0)
            cam_info = self.get_camera_info(color_image, depth_image, timestamp)
            self.frame_id += 1
            return cam_info
        except queue.Empty:
            raise StopIteration("No frames received in 5 seconds")

    def __enter__(self):
        return self.start()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
        return False

