#
# Network Streaming Module for RTG-SLAM
# TCP-based client-server communication for remote SLAM
#

import socket
import struct
import threading
import queue
import time
import numpy as np
from PIL import Image
from typing import Optional, Tuple, Callable

from scene.dataset_readers import CameraInfo
from utils.graphics_utils import focal2fov


class NetworkProtocol:
    """Simple binary protocol for image transmission"""

    # Message types
    MSG_FRAME = 1        # Client -> Server: RGB + Depth frame
    MSG_RENDER = 2       # Server -> Client: Rendered result
    MSG_INTRINSICS = 3   # Client -> Server: Camera intrinsics
    MSG_STATUS = 4       # Server -> Client: Status update
    MSG_SHUTDOWN = 5     # Client -> Server: Shutdown request

    HEADER_FORMAT = '!BIIII'  # msg_type(1), frame_id(4), timestamp_ms(4), width(4), height(4)
    HEADER_SIZE = struct.calcsize(HEADER_FORMAT)

    @staticmethod
    def pack_header(msg_type: int, frame_id: int, timestamp: float, width: int, height: int) -> bytes:
        timestamp_ms = int(timestamp * 1000)
        return struct.pack(NetworkProtocol.HEADER_FORMAT, msg_type, frame_id, timestamp_ms, width, height)

    @staticmethod
    def unpack_header(data: bytes) -> Tuple[int, int, float, int, int]:
        msg_type, frame_id, timestamp_ms, width, height = struct.unpack(NetworkProtocol.HEADER_FORMAT, data)
        return msg_type, frame_id, timestamp_ms / 1000.0, width, height

    @staticmethod
    def pack_intrinsics(fx: float, fy: float, cx: float, cy: float) -> bytes:
        return struct.pack('!ffff', fx, fy, cx, cy)

    @staticmethod
    def unpack_intrinsics(data: bytes) -> Tuple[float, float, float, float]:
        return struct.unpack('!ffff', data)

    @staticmethod
    def pack_pose(pose: np.ndarray) -> bytes:
        """Pack 4x4 pose matrix"""
        return pose.astype(np.float32).tobytes()

    @staticmethod
    def unpack_pose(data: bytes) -> np.ndarray:
        """Unpack 4x4 pose matrix"""
        return np.frombuffer(data, dtype=np.float32).reshape(4, 4)


def recv_exact(sock: socket.socket, size: int) -> bytes:
    """Receive exactly `size` bytes from socket"""
    data = b''
    while len(data) < size:
        chunk = sock.recv(size - len(data))
        if not chunk:
            raise ConnectionError("Connection closed")
        data += chunk
    return data


class NetworkStreamServer:
    """
    Server that receives RGB-D frames from network clients.
    Compatible with RTG-SLAM's RealsenseStream interface.
    """

    def __init__(
        self,
        host: str = '0.0.0.0',
        port: int = 9999,
        queue_size: int = 2,
    ):
        self.host = host
        self.port = port
        self.queue_size = queue_size

        # Server socket
        self.server_socket = None
        self.client_socket = None
        self.client_address = None

        # Threading
        self.frame_queue = queue.Queue(maxsize=queue_size)
        self.receive_thread = None
        self.running = False

        # Frame counter
        self.frame_id = 0
        self.start_time = None

        # Camera intrinsics (set by client)
        self.fx = 0
        self.fy = 0
        self.cx = 0
        self.cy = 0
        self.width = 0
        self.height = 0
        self.intrinsics_received = False

        # Callback for sending rendered results
        self.render_callback: Optional[Callable] = None

    def start(self):
        """Start server and wait for client connection"""
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server_socket.bind((self.host, self.port))
        self.server_socket.listen(1)

        print(f"[NetworkStreamServer] Listening on {self.host}:{self.port}")
        print("[NetworkStreamServer] Waiting for client connection...")

        self.client_socket, self.client_address = self.server_socket.accept()
        print(f"[NetworkStreamServer] Client connected: {self.client_address}")

        # Wait for intrinsics first
        self._receive_intrinsics()

        # Start receive thread
        self.running = True
        self.start_time = time.time()
        self.receive_thread = threading.Thread(target=self._receive_loop, daemon=True)
        self.receive_thread.start()

        print(f"[NetworkStreamServer] Started, resolution {self.width}x{self.height}")
        print(f"[NetworkStreamServer] Intrinsics: fx={self.fx:.2f}, fy={self.fy:.2f}, cx={self.cx:.2f}, cy={self.cy:.2f}")

        return self

    def _receive_intrinsics(self):
        """Receive camera intrinsics from client"""
        header = recv_exact(self.client_socket, NetworkProtocol.HEADER_SIZE)
        msg_type, _, _, width, height = NetworkProtocol.unpack_header(header)

        if msg_type != NetworkProtocol.MSG_INTRINSICS:
            raise ValueError(f"Expected intrinsics message, got {msg_type}")

        intrinsics_data = recv_exact(self.client_socket, 16)  # 4 floats
        self.fx, self.fy, self.cx, self.cy = NetworkProtocol.unpack_intrinsics(intrinsics_data)
        self.width = width
        self.height = height
        self.intrinsics_received = True

    def _receive_loop(self):
        """Background thread for receiving frames"""
        while self.running:
            try:
                # Receive header
                header = recv_exact(self.client_socket, NetworkProtocol.HEADER_SIZE)
                msg_type, frame_id, timestamp, width, height = NetworkProtocol.unpack_header(header)

                if msg_type == NetworkProtocol.MSG_SHUTDOWN:
                    print("[NetworkStreamServer] Shutdown request received")
                    self.running = False
                    break

                if msg_type != NetworkProtocol.MSG_FRAME:
                    continue

                # Receive RGB data (H x W x 3, uint8)
                rgb_size = width * height * 3
                rgb_data = recv_exact(self.client_socket, rgb_size)
                rgb_image = np.frombuffer(rgb_data, dtype=np.uint8).reshape(height, width, 3)

                # Receive depth data (H x W, float32)
                depth_size = width * height * 4
                depth_data = recv_exact(self.client_socket, depth_size)
                depth_image = np.frombuffer(depth_data, dtype=np.float32).reshape(height, width)

                # Put in queue
                try:
                    self.frame_queue.put_nowait((rgb_image, depth_image, timestamp, frame_id))
                except queue.Full:
                    # Drop oldest frame
                    try:
                        self.frame_queue.get_nowait()
                        self.frame_queue.put_nowait((rgb_image, depth_image, timestamp, frame_id))
                    except queue.Empty:
                        pass

            except ConnectionError:
                print("[NetworkStreamServer] Client disconnected")
                self.running = False
                break
            except Exception as e:
                if self.running:
                    print(f"[NetworkStreamServer] Receive error: {e}")

    def send_render_result(
        self,
        frame_id: int,
        timestamp: float,
        rendered_rgb: np.ndarray,
        rendered_depth: np.ndarray,
        camera_pose: np.ndarray,
    ):
        """Send rendered result back to client"""
        if not self.client_socket or not self.running:
            return

        try:
            height, width = rendered_rgb.shape[:2]

            # Pack header
            header = NetworkProtocol.pack_header(
                NetworkProtocol.MSG_RENDER, frame_id, timestamp, width, height
            )

            # Pack data
            rgb_bytes = rendered_rgb.astype(np.uint8).tobytes()
            depth_bytes = rendered_depth.astype(np.float32).tobytes()
            pose_bytes = NetworkProtocol.pack_pose(camera_pose)

            # Send all data
            self.client_socket.sendall(header + rgb_bytes + depth_bytes + pose_bytes)

        except Exception as e:
            print(f"[NetworkStreamServer] Send error: {e}")

    def stop(self):
        """Stop the server"""
        self.running = False
        if self.receive_thread:
            self.receive_thread.join(timeout=2.0)
        if self.client_socket:
            self.client_socket.close()
        if self.server_socket:
            self.server_socket.close()
        print("[NetworkStreamServer] Stopped")

    def get_camera_info(self, rgb_image: np.ndarray, depth_image: np.ndarray, timestamp: float) -> CameraInfo:
        """Convert raw frames to CameraInfo"""
        # Convert to PIL Image
        image_pil = Image.fromarray(rgb_image)
        depth_pil = Image.fromarray(depth_image)

        # Calculate FoV
        FovX = focal2fov(self.fx, self.width)
        FovY = focal2fov(self.fy, self.height)

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
            image_path=f"network_frame_{self.frame_id:06d}",
            image_name=f"frame_{self.frame_id:06d}",
            width=self.width,
            height=self.height,
            depth=depth_pil,
            depth_path=f"network_depth_{self.frame_id:06d}",
            pose_gt=pose_gt,
            cx=self.cx,
            cy=self.cy,
            depth_scale=1.0,
            timestamp=timestamp,
        )

    def __iter__(self):
        return self

    def __next__(self) -> CameraInfo:
        if not self.running:
            raise StopIteration

        try:
            rgb_image, depth_image, timestamp, _ = self.frame_queue.get(timeout=10.0)
            cam_info = self.get_camera_info(rgb_image, depth_image, timestamp)
            self.frame_id += 1
            return cam_info
        except queue.Empty:
            raise StopIteration("No frames received in 10 seconds")

    def __enter__(self):
        return self.start()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
        return False


class NetworkStreamClient:
    """
    Client that sends RGB-D frames to SLAM server and receives rendered results.
    """

    def __init__(
        self,
        host: str = 'localhost',
        port: int = 9999,
    ):
        self.host = host
        self.port = port
        self.socket = None
        self.connected = False

        # Receive thread
        self.receive_thread = None
        self.running = False

        # Callback for received renders
        self.on_render_received: Optional[Callable] = None

        # Frame tracking
        self.frame_id = 0
        self.start_time = None

    def connect(self, fx: float, fy: float, cx: float, cy: float, width: int, height: int):
        """Connect to server and send camera intrinsics"""
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.connect((self.host, self.port))
        self.connected = True
        self.start_time = time.time()

        print(f"[NetworkStreamClient] Connected to {self.host}:{self.port}")

        # Send intrinsics
        header = NetworkProtocol.pack_header(NetworkProtocol.MSG_INTRINSICS, 0, 0, width, height)
        intrinsics = NetworkProtocol.pack_intrinsics(fx, fy, cx, cy)
        self.socket.sendall(header + intrinsics)

        # Start receive thread
        self.running = True
        self.receive_thread = threading.Thread(target=self._receive_loop, daemon=True)
        self.receive_thread.start()

        print("[NetworkStreamClient] Intrinsics sent, ready to stream")
        return self

    def _receive_loop(self):
        """Background thread for receiving rendered results"""
        while self.running:
            try:
                # Receive header
                header = recv_exact(self.socket, NetworkProtocol.HEADER_SIZE)
                msg_type, frame_id, timestamp, width, height = NetworkProtocol.unpack_header(header)

                if msg_type != NetworkProtocol.MSG_RENDER:
                    continue

                # Receive rendered RGB
                rgb_size = width * height * 3
                rgb_data = recv_exact(self.socket, rgb_size)
                rendered_rgb = np.frombuffer(rgb_data, dtype=np.uint8).reshape(height, width, 3)

                # Receive rendered depth
                depth_size = width * height * 4
                depth_data = recv_exact(self.socket, depth_size)
                rendered_depth = np.frombuffer(depth_data, dtype=np.float32).reshape(height, width)

                # Receive camera pose
                pose_data = recv_exact(self.socket, 64)  # 4x4 float32
                camera_pose = NetworkProtocol.unpack_pose(pose_data)

                # Call callback if set
                if self.on_render_received:
                    self.on_render_received(frame_id, timestamp, rendered_rgb, rendered_depth, camera_pose)

            except ConnectionError:
                print("[NetworkStreamClient] Server disconnected")
                self.running = False
                break
            except Exception as e:
                if self.running:
                    print(f"[NetworkStreamClient] Receive error: {e}")

    def send_frame(self, rgb_image: np.ndarray, depth_image: np.ndarray, timestamp: Optional[float] = None):
        """Send RGB-D frame to server"""
        if not self.connected:
            raise RuntimeError("Not connected to server")

        if timestamp is None:
            timestamp = time.time() - self.start_time

        height, width = rgb_image.shape[:2]

        # Pack header
        header = NetworkProtocol.pack_header(
            NetworkProtocol.MSG_FRAME, self.frame_id, timestamp, width, height
        )

        # Pack image data
        rgb_bytes = rgb_image.astype(np.uint8).tobytes()
        depth_bytes = depth_image.astype(np.float32).tobytes()

        # Send all data
        self.socket.sendall(header + rgb_bytes + depth_bytes)

        self.frame_id += 1

    def shutdown_server(self):
        """Send shutdown request to server"""
        if self.connected:
            header = NetworkProtocol.pack_header(NetworkProtocol.MSG_SHUTDOWN, 0, 0, 0, 0)
            self.socket.sendall(header)

    def disconnect(self):
        """Disconnect from server"""
        self.running = False
        if self.receive_thread:
            self.receive_thread.join(timeout=2.0)
        if self.socket:
            self.socket.close()
        self.connected = False
        print("[NetworkStreamClient] Disconnected")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.disconnect()
        return False