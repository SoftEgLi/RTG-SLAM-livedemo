"""
RTG-SLAM Server - Receives RGB-D images via network and performs real-time SLAM

Usage:
    python slam_server.py --config configs/network/server.yaml --port 9999
"""

import os
import signal
import sys
import time
import json
from argparse import ArgumentParser

from utils.config_utils import read_config

parser = ArgumentParser(description="RTG-SLAM Network Server")
parser.add_argument("--config", type=str, default="configs/realsense/d455.yaml")
parser.add_argument("--host", type=str, default="0.0.0.0", help="Server host address")
parser.add_argument("--port", type=int, default=9999, help="Server port")
parser.add_argument("--max_frames", type=int, default=-1, help="Max frames to process (-1 for unlimited)")
parser.add_argument("--visualize", action="store_true", help="Enable visualization window")
parser.add_argument("--save_data", action="store_true", help="Save RGB, depth images and poses")
parser.add_argument("--save_data_dir", type=str, default="network_slam/my_experiment", help="Directory to save data (default: save_path/recorded_data)")
args = parser.parse_args()
config_path = args.config
args_config = read_config(config_path)

# Merge command line args
args_config.max_frames = args.max_frames
args_config.visualize = args.visualize
args_config.save_data = args.save_data
args_config.save_data_dir = args.save_data_dir

os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(device) for device in args_config.device_list)

import cv2
import numpy as np
import torch
from utils.camera_utils import loadCam
from arguments import DatasetParams, MapParams, OptimizationParams
from scene.network_stream import NetworkStreamServer
from SLAM.multiprocess.mapper import Mapping
from SLAM.multiprocess.tracker import Tracker
from SLAM.utils import *
from SLAM.eval import eval_frame
from utils.general_utils import safe_state
from utils.monitor import Recorder

torch.set_printoptions(4, sci_mode=False)

# Global flag for graceful shutdown
shutdown_requested = False


def signal_handler(sig, frame):
    global shutdown_requested
    print("\n[SIGNAL] Shutdown requested, finishing current frame...")
    shutdown_requested = True


def depth_to_colormap(depth, min_depth=0.0, max_depth=5.0):
    """Convert depth tensor to colormap image for visualization."""
    if isinstance(depth, torch.Tensor):
        depth = depth.detach().cpu().numpy()

    if depth.ndim == 3:
        if depth.shape[0] == 1:
            depth = depth[0]
        elif depth.shape[2] == 1:
            depth = depth[:, :, 0]

    depth = np.clip(depth, min_depth, max_depth)
    depth_normalized = ((depth - min_depth) / (max_depth - min_depth) * 255).astype(np.uint8)
    depth_colormap = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_JET)
    return depth_colormap


class DataRecorder:
    """Class to save RGB images, depth images and poses during SLAM."""

    def __init__(self, save_dir):
        self.save_dir = save_dir
        self.rgb_dir = os.path.join(save_dir, "rgb")
        self.depth_dir = os.path.join(save_dir, "depth")
        self.pose_dir = os.path.join(save_dir, "pose")

        # Create directories
        os.makedirs(self.rgb_dir, exist_ok=True)
        os.makedirs(self.depth_dir, exist_ok=True)
        os.makedirs(self.pose_dir, exist_ok=True)

        print(f"[DataRecorder] Saving data to: {save_dir}")
        print(f"  - RGB images: {self.rgb_dir}")
        print(f"  - Depth images: {self.depth_dir}")
        print(f"  - Poses: {self.pose_dir}")

    def save_frame(self, frame_id, curr_frame, timestamp):
        """Save RGB, depth and pose for a single frame."""
        # Save RGB image (shape: 3, H, W -> H, W, 3)
        rgb = curr_frame.original_image.detach().cpu().numpy()
        rgb = np.transpose(rgb, (1, 2, 0))  # (H, W, 3)
        rgb = (rgb * 255).astype(np.uint8)
        rgb_bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        rgb_path = os.path.join(self.rgb_dir, f"{frame_id:06d}.png")
        cv2.imwrite(rgb_path, rgb_bgr)

        # Save depth image (shape: 1, H, W) as 16-bit PNG (millimeters)
        depth = curr_frame.original_depth.detach().cpu().numpy()
        if depth.ndim == 3:
            if depth.shape[0] == 1:
                depth = depth[0]
            elif depth.shape[2] == 1:
                depth = depth[:, :, 0]
        # Convert to millimeters and save as 16-bit
        depth_mm = (depth * 1000).astype(np.uint16)
        depth_path = os.path.join(self.depth_dir, f"{frame_id:06d}.png")
        cv2.imwrite(depth_path, depth_mm)

        # Save pose as 4x4 matrix (same format as 0.txt)
        c2w = curr_frame.get_c2w.detach().cpu().numpy()  # 4x4 matrix
        pose_path = os.path.join(self.pose_dir, f"{frame_id:06d}.txt")
        with open(pose_path, "w") as f:
            for row in c2w:
                f.write(" ".join(f"{v:.6f}" for v in row) + "\n")

    def save_camera_intrinsics(self, curr_frame):
        """Save camera intrinsic parameters."""
        intrinsics_file = os.path.join(self.save_dir, "camera_intrinsics.txt")

        # Get intrinsic matrix from Camera
        intrinsic = curr_frame.get_intrinsic.cpu().numpy()
        fx = intrinsic[0, 0]
        fy = intrinsic[1, 1]
        cx = intrinsic[0, 2]
        cy = intrinsic[1, 2]
        width = curr_frame.image_width
        height = curr_frame.image_height

        with open(intrinsics_file, "w") as f:
            f.write(f"fx: {fx}\n")
            f.write(f"fy: {fy}\n")
            f.write(f"cx: {cx}\n")
            f.write(f"cy: {cy}\n")
            f.write(f"width: {width}\n")
            f.write(f"height: {height}\n")

        print(f"[DataRecorder] Saved camera intrinsics to: {intrinsics_file}")


def save_cameras_json(keyframe_list, save_path):
    """Save camera intrinsics and poses to cameras.json"""
    if not keyframe_list:
        print("[Warning] No keyframes to save in cameras.json")
        return

    cameras = []
    for idx, frame in enumerate(keyframe_list):
        # Get pose (camera-to-world)
        c2w = frame.get_c2w.detach().cpu().numpy()

        # Skip invalid poses
        if np.isinf(c2w).any() or np.isnan(c2w).any():
            print(f"[Warning] Skipping invalid pose at frame {idx}")
            continue

        R = c2w[:3, :3]
        T = c2w[:3, 3]

        # Get intrinsics
        intrinsic = frame.get_intrinsic.cpu().numpy()
        fx = float(intrinsic[0, 0])
        fy = float(intrinsic[1, 1])

        cameras.append({
            "id": idx,
            "img_name": f"frame_{idx:06d}",
            "width": frame.image_width,
            "height": frame.image_height,
            "position": T.tolist(),
            "rotation": [row.tolist() for row in R],
            "fx": fx,
            "fy": fy,
        })

    cameras_path = os.path.join(save_path, "cameras.json")
    with open(cameras_path, "w") as f:
        json.dump(cameras, f, indent=2)
    print(f"[LOG] Saved cameras.json with {len(cameras)} cameras to {cameras_path}")


def create_visualization(curr_frame, gaussian_map, min_depth=0.0, max_depth=5.0):
    """Create a 2x2 visualization grid showing camera and rendered images."""
    # Get camera captured RGB image (shape: 3, H, W)
    camera_rgb = curr_frame.original_image.detach().cpu().numpy()
    camera_rgb = np.transpose(camera_rgb, (1, 2, 0))
    camera_rgb = (camera_rgb * 255).astype(np.uint8)
    camera_rgb = cv2.cvtColor(camera_rgb, cv2.COLOR_RGB2BGR)

    # Get camera captured depth
    camera_depth = curr_frame.original_depth.detach().cpu().numpy()
    camera_depth_colormap = depth_to_colormap(camera_depth, min_depth, max_depth)

    # Get rendered RGB
    render_rgb = gaussian_map.model_map["render_color"]
    if isinstance(render_rgb, torch.Tensor):
        render_rgb = render_rgb.detach().cpu().numpy()
    render_rgb = (np.clip(render_rgb, 0, 1) * 255).astype(np.uint8)
    render_rgb = cv2.cvtColor(render_rgb, cv2.COLOR_RGB2BGR)

    # Get rendered depth
    render_depth = gaussian_map.model_map["render_depth"]
    render_depth_colormap = depth_to_colormap(render_depth, min_depth, max_depth)

    h, w = camera_rgb.shape[:2]

    # Add labels
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    font_color = (255, 255, 255)
    font_thickness = 2
    bg_color = (0, 0, 0)

    def add_label(img, label):
        img_copy = img.copy()
        (text_width, text_height), _ = cv2.getTextSize(label, font, font_scale, font_thickness)
        cv2.rectangle(img_copy, (5, 5), (15 + text_width, 15 + text_height), bg_color, -1)
        cv2.putText(img_copy, label, (10, 10 + text_height), font, font_scale, font_color, font_thickness)
        return img_copy

    camera_rgb_labeled = add_label(camera_rgb, "Input RGB")
    camera_depth_labeled = add_label(camera_depth_colormap, "Input Depth")
    render_rgb_labeled = add_label(render_rgb, "Rendered RGB")
    render_depth_labeled = add_label(render_depth_colormap, "Rendered Depth")

    top_row = np.hstack([camera_rgb_labeled, camera_depth_labeled])
    bottom_row = np.hstack([render_rgb_labeled, render_depth_labeled])
    grid = np.vstack([top_row, bottom_row])

    return grid


def main():
    global shutdown_requested

    # Set multiprocessing start method
    import torch.multiprocessing as mp
    mp.set_start_method('spawn', force=True)

    # Register signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    time_recorder = Recorder(args_config.device_list[0])
    optimization_params = OptimizationParams(parser)
    dataset_params = DatasetParams(parser, sentinel=True)
    map_params = MapParams(parser)

    safe_state(args_config.quiet)
    optimization_params = optimization_params.extract(args_config)
    dataset_params = dataset_params.extract(args_config)
    map_params = map_params.extract(args_config)

    record_mem = args_config.record_mem

    gaussian_map = Mapping(args_config, time_recorder)
    gaussian_map.create_workspace()
    gaussian_tracker = Tracker(args_config)

    # Save config file
    prepare_cfg(args_config)

    # Initialize data recorder if save_data is enabled
    data_recorder = None
    if args_config.save_data:
        save_data_dir = args_config.save_data_dir
        if save_data_dir is None:
            save_data_dir = os.path.join(args_config.save_path, "recorded_data")
        data_recorder = DataRecorder(save_data_dir)

    tracker_time_sum = 0
    mapper_time_sum = 0
    frame_id = 0

    # Start network server
    print("\n========== Starting RTG-SLAM Network Server ==========\n")
    print(f"Host: {args.host}, Port: {args.port}")
    print("Waiting for client connection...")

    with NetworkStreamServer(host=args.host, port=args.port) as stream:
        for frame_info in stream:
            if shutdown_requested:
                print("[INFO] Shutdown requested, stopping...")
                break

            if args_config.max_frames > 0 and frame_id >= args_config.max_frames:
                print(f"[INFO] Reached max frames ({args_config.max_frames}), stopping...")
                break

            # Convert CameraInfo to Camera
            curr_frame = loadCam(
                dataset_params, frame_id, frame_info, dataset_params.resolution_scales[0]
            )

            print(f"\n========== Processing frame: {frame_id} (t={frame_info.timestamp:.3f}s) ==========\n")
            move_to_gpu(curr_frame)
            start_time = time.time()

            # Tracker process
            frame_map = gaussian_tracker.map_preprocess(curr_frame, frame_id)
            gaussian_tracker.tracking(curr_frame, frame_map)
            tracker_time = time.time()
            tracker_consume_time = tracker_time - start_time
            time_recorder.update_mean("tracking", tracker_consume_time, 1)

            tracker_time_sum += tracker_consume_time
            print(f"[LOG] tracker cost time: {tracker_time - start_time:.4f}s")

            mapper_start_time = time.time()

            new_poses = gaussian_tracker.get_new_poses()
            gaussian_map.update_poses(new_poses)

            # Mapper process
            gaussian_map.mapping(curr_frame, frame_map, frame_id, optimization_params)

            gaussian_map.get_render_output(curr_frame)
            gaussian_tracker.update_last_status(
                curr_frame,
                gaussian_map.model_map["render_depth"],
                gaussian_map.frame_map["depth_map"],
                gaussian_map.model_map["render_normal"],
                gaussian_map.frame_map["normal_map_w"],
            )
            mapper_time = time.time()
            mapper_consume_time = mapper_time - mapper_start_time
            time_recorder.update_mean("mapping", mapper_consume_time, 1)

            mapper_time_sum += mapper_consume_time
            print(f"[LOG] mapper cost time: {mapper_time - tracker_time:.4f}s")
            print(f"[LOG] total frame time: {mapper_time - start_time:.4f}s ({1.0/(mapper_time - start_time):.1f} FPS)")

            # Get rendered results to send back to client
            render_rgb = gaussian_map.model_map["render_color"]
            if isinstance(render_rgb, torch.Tensor):
                render_rgb = render_rgb.detach().cpu().numpy()
            render_rgb = (np.clip(render_rgb, 0, 1) * 255).astype(np.uint8)

            render_depth = gaussian_map.model_map["render_depth"]
            if isinstance(render_depth, torch.Tensor):
                render_depth = render_depth.detach().cpu().numpy()
            if render_depth.ndim == 3:
                if render_depth.shape[2] == 1:
                    render_depth = render_depth[:, :, 0]

            camera_pose = curr_frame.get_c2w.detach().cpu().numpy()

            # Send rendered result back to client
            stream.send_render_result(
                frame_id=frame_id,
                timestamp=frame_info.timestamp,
                rendered_rgb=render_rgb,
                rendered_depth=render_depth,
                camera_pose=camera_pose,
            )

            # Save data (RGB, depth, pose)
            if data_recorder is not None:
                if frame_id == 0:
                    data_recorder.save_camera_intrinsics(curr_frame)
                data_recorder.save_frame(frame_id, curr_frame, frame_info.timestamp)

            # Visualization
            if args_config.visualize:
                vis_grid = create_visualization(
                    curr_frame, gaussian_map,
                    min_depth=gaussian_map.min_depth,
                    max_depth=gaussian_map.max_depth
                )
                cv2.imshow("RTG-SLAM Server", vis_grid)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == 27:
                    print("[INFO] Quit requested via visualization window...")
                    shutdown_requested = True

            if record_mem:
                time_recorder.watch_gpu()

            # Periodic evaluation and saving
            if ((gaussian_map.time + 1) % gaussian_map.save_step == 0) or (gaussian_map.time == 0):
                eval_frame(
                    gaussian_map,
                    curr_frame,
                    os.path.join(gaussian_map.save_path, "eval_render"),
                    min_depth=gaussian_map.min_depth,
                    max_depth=gaussian_map.max_depth,
                    save_picture=True,
                    run_pcd=False
                )
                gaussian_map.save_model(save_data=True)

            gaussian_map.time += 1
            frame_id += 1
            move_to_cpu(curr_frame)
            torch.cuda.empty_cache()

    # Finalization
    print("\n========== Main loop finished ==========\n")
    print(f"[LOG] Total frames processed: {frame_id}")
    print(f"[LOG] stable num: {gaussian_map.get_stable_num}, unstable num: {gaussian_map.get_unstable_num}")
    print(f"[LOG] processed frames: {gaussian_map.optimize_frames_ids}")
    print(f"[LOG] keyframes: {gaussian_map.keyframe_ids}")

    if frame_id > 0:
        print(f"[LOG] mean tracker process time: {tracker_time_sum / frame_id:.4f}s")
        print(f"[LOG] mean mapper process time: {mapper_time_sum / frame_id:.4f}s")

    # Global optimization
    new_poses = gaussian_tracker.get_new_poses()
    gaussian_map.update_poses(new_poses)
    gaussian_map.global_optimization(optimization_params, is_end=True)

    if gaussian_map.keyframe_list:
        eval_frame(
            gaussian_map,
            gaussian_map.keyframe_list[-1],
            os.path.join(gaussian_map.save_path, "eval_render"),
            min_depth=gaussian_map.min_depth,
            max_depth=gaussian_map.max_depth,
            save_picture=True,
            run_pcd=False
        )

    gaussian_map.save_model(save_data=True)
    gaussian_tracker.save_traj(args_config.save_path)
    time_recorder.cal_fps()
    time_recorder.save(args_config.save_path)
    gaussian_map.time += 1

    # Save cameras.json
    save_cameras_json(gaussian_map.keyframe_list, args_config.save_path)

    if args_config.pcd_densify:
        densify_pcd = gaussian_map.stable_pointcloud.densify(1, 30, 5)
        o3d.io.write_point_cloud(
            os.path.join(args_config.save_path, "save_model", "pcd_densify.ply"), densify_pcd
        )

    print("\n========== SLAM Server Finished ==========\n")

    if args_config.visualize:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()