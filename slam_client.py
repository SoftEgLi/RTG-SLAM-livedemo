"""
RTG-SLAM Client - Sends RGB-D images to SLAM server and receives rendered results

Usage examples:
    # Send from RealSense camera
    python slam_client.py --mode realsense --host localhost --port 9999

    # Send from image folder
    python slam_client.py --mode folder --data_path /path/to/dataset --host localhost --port 9999

    # Send from video files
    python slam_client.py --mode video --rgb_video rgb.mp4 --depth_video depth.mp4 --host localhost
"""

import argparse
import os
import time
import threading
import queue
import numpy as np
import cv2
from typing import Optional

from scene.network_stream import NetworkStreamClient


class RealSenseSource:
    """RGB-D source from RealSense camera"""

    def __init__(self, width=640, height=480, fps=30):
        import pyrealsense2 as rs

        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.color, width, height, rs.format.rgb8, fps)
        config.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)

        self.profile = self.pipeline.start(config)

        # Get depth scale
        depth_sensor = self.profile.get_device().first_depth_sensor()
        self.depth_scale = depth_sensor.get_depth_scale()

        # Align depth to color
        self.align = rs.align(rs.stream.color)

        # Get intrinsics
        color_stream = self.profile.get_stream(rs.stream.color)
        intrinsics = color_stream.as_video_stream_profile().get_intrinsics()
        self.fx = intrinsics.fx
        self.fy = intrinsics.fy
        self.cx = intrinsics.ppx
        self.cy = intrinsics.ppy
        self.width = intrinsics.width
        self.height = intrinsics.height

        print(f"[RealSenseSource] Started {self.width}x{self.height}")
        print(f"[RealSenseSource] Intrinsics: fx={self.fx:.2f}, fy={self.fy:.2f}")

    def get_frame(self):
        frames = self.pipeline.wait_for_frames()
        frames = self.align.process(frames)

        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()

        if not color_frame or not depth_frame:
            return None, None

        rgb = np.asanyarray(color_frame.get_data())
        depth = np.asanyarray(depth_frame.get_data()).astype(np.float32) * self.depth_scale

        return rgb, depth

    def stop(self):
        self.pipeline.stop()


class FolderSource:
    """RGB-D source from image folder (TUM/Replica format)"""

    def __init__(self, data_path: str, fx: float, fy: float, cx: float, cy: float):
        self.data_path = data_path
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy

        # Try to find RGB and depth directories
        self.rgb_dir = None
        self.depth_dir = None

        for rgb_name in ['rgb', 'color', 'images']:
            path = os.path.join(data_path, rgb_name)
            if os.path.exists(path):
                self.rgb_dir = path
                break

        for depth_name in ['depth', 'depths']:
            path = os.path.join(data_path, depth_name)
            if os.path.exists(path):
                self.depth_dir = path
                break

        if not self.rgb_dir or not self.depth_dir:
            raise ValueError(f"Cannot find rgb/depth directories in {data_path}")

        # Get sorted file lists
        self.rgb_files = sorted([f for f in os.listdir(self.rgb_dir) if f.endswith(('.png', '.jpg'))])
        self.depth_files = sorted([f for f in os.listdir(self.depth_dir) if f.endswith('.png')])

        if len(self.rgb_files) != len(self.depth_files):
            print(f"Warning: RGB ({len(self.rgb_files)}) and depth ({len(self.depth_files)}) counts differ")

        self.num_frames = min(len(self.rgb_files), len(self.depth_files))
        self.current_idx = 0

        # Get image size from first frame
        first_rgb = cv2.imread(os.path.join(self.rgb_dir, self.rgb_files[0]))
        self.height, self.width = first_rgb.shape[:2]

        print(f"[FolderSource] Found {self.num_frames} frames at {self.width}x{self.height}")

    def get_frame(self):
        if self.current_idx >= self.num_frames:
            return None, None

        rgb_path = os.path.join(self.rgb_dir, self.rgb_files[self.current_idx])
        depth_path = os.path.join(self.depth_dir, self.depth_files[self.current_idx])

        rgb = cv2.imread(rgb_path)
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)

        depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
        if depth.dtype == np.uint16:
            depth = depth.astype(np.float32) / 1000.0  # mm to meters
        else:
            depth = depth.astype(np.float32)

        self.current_idx += 1
        return rgb, depth

    def stop(self):
        pass


class VideoSource:
    """RGB-D source from video files"""

    def __init__(self, rgb_video: str, depth_video: str, fx: float, fy: float, cx: float, cy: float):
        self.rgb_cap = cv2.VideoCapture(rgb_video)
        self.depth_cap = cv2.VideoCapture(depth_video)

        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy

        self.width = int(self.rgb_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.rgb_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        print(f"[VideoSource] Opened videos at {self.width}x{self.height}")

    def get_frame(self):
        ret_rgb, rgb = self.rgb_cap.read()
        ret_depth, depth = self.depth_cap.read()

        if not ret_rgb or not ret_depth:
            return None, None

        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)

        # Assume depth is stored as 16-bit grayscale in mm
        if len(depth.shape) == 3:
            depth = cv2.cvtColor(depth, cv2.COLOR_BGR2GRAY)
        depth = depth.astype(np.float32) / 1000.0

        return rgb, depth

    def stop(self):
        self.rgb_cap.release()
        self.depth_cap.release()


def depth_to_colormap(depth, min_depth=0.0, max_depth=5.0):
    """Convert depth to colormap for visualization"""
    depth = np.clip(depth, min_depth, max_depth)
    depth_normalized = ((depth - min_depth) / (max_depth - min_depth) * 255).astype(np.uint8)
    return cv2.applyColorMap(depth_normalized, cv2.COLORMAP_JET)


def main():
    parser = argparse.ArgumentParser(description="RTG-SLAM Network Client")
    parser.add_argument("--mode", type=str, choices=['realsense', 'folder', 'video'], default='realsense')
    parser.add_argument("--host", type=str, default="localhost", help="Server host")
    parser.add_argument("--port", type=int, default=9999, help="Server port")

    # RealSense options
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)

    # Folder/Video options
    parser.add_argument("--data_path", type=str, help="Path to dataset folder")
    parser.add_argument("--rgb_video", type=str, help="Path to RGB video")
    parser.add_argument("--depth_video", type=str, help="Path to depth video")

    # Camera intrinsics (for folder/video mode)
    parser.add_argument("--fx", type=float, default=600.0)
    parser.add_argument("--fy", type=float, default=600.0)
    parser.add_argument("--cx", type=float, default=320.0)
    parser.add_argument("--cy", type=float, default=240.0)

    # Other options
    parser.add_argument("--max_frames", type=int, default=-1)
    parser.add_argument("--frame_delay", type=float, default=0.0, help="Delay between frames (seconds)")
    parser.add_argument("--visualize", action="store_true", help="Show visualization")

    args = parser.parse_args()

    # Create data source
    if args.mode == 'realsense':
        source = RealSenseSource(args.width, args.height, args.fps)
    elif args.mode == 'folder':
        if not args.data_path:
            raise ValueError("--data_path required for folder mode")
        source = FolderSource(args.data_path, args.fx, args.fy, args.cx, args.cy)
    elif args.mode == 'video':
        if not args.rgb_video or not args.depth_video:
            raise ValueError("--rgb_video and --depth_video required for video mode")
        source = VideoSource(args.rgb_video, args.depth_video, args.fx, args.fy, args.cx, args.cy)
    else:
        raise ValueError(f"Unknown mode: {args.mode}")

    # Storage for received renders
    render_queue = queue.Queue(maxsize=2)

    def on_render_received(frame_id, timestamp, rendered_rgb, rendered_depth, camera_pose):
        """Callback when rendered result is received from server"""
        try:
            render_queue.put_nowait((frame_id, timestamp, rendered_rgb, rendered_depth, camera_pose))
        except queue.Full:
            try:
                render_queue.get_nowait()
                render_queue.put_nowait((frame_id, timestamp, rendered_rgb, rendered_depth, camera_pose))
            except queue.Empty:
                pass

    # Connect to server
    client = NetworkStreamClient(host=args.host, port=args.port)
    client.on_render_received = on_render_received
    client.connect(source.fx, source.fy, source.cx, source.cy, source.width, source.height)

    print("\n========== RTG-SLAM Client Started ==========\n")
    print(f"Connected to server at {args.host}:{args.port}")
    print("Press 'q' to quit\n")

    frame_count = 0
    start_time = time.time()

    try:
        while True:
            # Check max frames
            if args.max_frames > 0 and frame_count >= args.max_frames:
                print(f"Reached max frames ({args.max_frames})")
                break

            # Get frame from source
            rgb, depth = source.get_frame()
            if rgb is None:
                print("No more frames from source")
                break

            # Send to server
            client.send_frame(rgb, depth)
            frame_count += 1

            elapsed = time.time() - start_time
            fps = frame_count / elapsed if elapsed > 0 else 0
            print(f"\r[Client] Sent frame {frame_count}, FPS: {fps:.1f}", end="", flush=True)

            # Visualization
            if args.visualize:
                # Show input
                input_rgb = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                input_depth = depth_to_colormap(depth)

                # Try to get rendered result
                try:
                    _, _, rendered_rgb, rendered_depth, pose = render_queue.get_nowait()
                    render_rgb_vis = cv2.cvtColor(rendered_rgb, cv2.COLOR_RGB2BGR)
                    render_depth_vis = depth_to_colormap(rendered_depth)
                except queue.Empty:
                    # No render result yet, show placeholder
                    render_rgb_vis = np.zeros_like(input_rgb)
                    render_depth_vis = np.zeros_like(input_depth)

                # Create 2x2 grid
                top_row = np.hstack([input_rgb, input_depth])
                bottom_row = np.hstack([render_rgb_vis, render_depth_vis])
                grid = np.vstack([top_row, bottom_row])

                # Add labels
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(grid, "Input RGB", (10, 25), font, 0.6, (255, 255, 255), 2)
                cv2.putText(grid, "Input Depth", (source.width + 10, 25), font, 0.6, (255, 255, 255), 2)
                cv2.putText(grid, "Rendered RGB", (10, source.height + 25), font, 0.6, (255, 255, 255), 2)
                cv2.putText(grid, "Rendered Depth", (source.width + 10, source.height + 25), font, 0.6, (255, 255, 255), 2)

                cv2.imshow("RTG-SLAM Client", grid)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == 27:
                    break

            # Frame delay
            if args.frame_delay > 0:
                time.sleep(args.frame_delay)

    except KeyboardInterrupt:
        print("\n[Client] Interrupted by user")
    finally:
        print(f"\n\n[Client] Sent {frame_count} frames in {time.time() - start_time:.1f}s")

        # Shutdown
        client.shutdown_server()
        client.disconnect()
        source.stop()

        if args.visualize:
            cv2.destroyAllWindows()

        print("[Client] Finished")


if __name__ == "__main__":
    main()