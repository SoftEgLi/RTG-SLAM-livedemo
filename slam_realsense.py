"""
RTG-SLAM with RealSense D455 Real-time Streaming

Usage:
    python slam_realsense.py --config configs/realsense/d455.yaml
"""

import os
import signal
import sys
from argparse import ArgumentParser

from utils.config_utils import read_config

parser = ArgumentParser(description="RTG-SLAM with RealSense streaming")
parser.add_argument("--config", type=str, default="configs/realsense/d455.yaml")
parser.add_argument("--width", type=int, default=640, help="Camera resolution width")
parser.add_argument("--height", type=int, default=480, help="Camera resolution height")
parser.add_argument("--fps", type=int, default=30, help="Camera frame rate")
parser.add_argument("--max_frames", type=int, default=-1, help="Max frames to process (-1 for unlimited)")
parser.add_argument("--visualize", action="store_true", help="Enable visualization window")
parser.add_argument("--save_data", action="store_true", help="Save RGB, depth images and poses")
parser.add_argument("--save_data_dir", type=str, default=None, help="Directory to save data (default: save_path/recorded_data)")
args = parser.parse_args()
config_path = args.config
args_config = read_config(config_path)

# Merge command line args
args_config.realsense_width = args.width
args_config.realsense_height = args.height
args_config.realsense_fps = args.fps
args_config.max_frames = args.max_frames
args_config.visualize = args.visualize

os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(device) for device in args_config.device_list)

import cv2
import numpy as np
import torch
import json
from utils.camera_utils import loadCam
from arguments import DatasetParams, MapParams, OptimizationParams
from scene.realsense_stream import RealsenseStream, RealsenseConfig
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

    # Handle different shapes
    if depth.ndim == 3:
        if depth.shape[0] == 1:  # (1, H, W)
            depth = depth[0]
        elif depth.shape[2] == 1:  # (H, W, 1)
            depth = depth[:, :, 0]

    # Normalize depth to 0-255
    depth = np.clip(depth, min_depth, max_depth)
    depth_normalized = ((depth - min_depth) / (max_depth - min_depth) * 255).astype(np.uint8)

    # Apply colormap
    depth_colormap = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_JET)
    return depth_colormap


def create_visualization(curr_frame, gaussian_map, min_depth=0.0, max_depth=5.0):
    """Create a 2x2 visualization grid showing camera and rendered images."""
    # Get camera captured RGB image (shape: 3, H, W)
    camera_rgb = curr_frame.original_image.detach().cpu().numpy()
    camera_rgb = np.transpose(camera_rgb, (1, 2, 0))  # (H, W, 3)
    camera_rgb = (camera_rgb * 255).astype(np.uint8)
    camera_rgb = cv2.cvtColor(camera_rgb, cv2.COLOR_RGB2BGR)

    # Get camera captured depth image (shape: 1, H, W)
    camera_depth = curr_frame.original_depth.detach().cpu().numpy()
    camera_depth_colormap = depth_to_colormap(camera_depth, min_depth, max_depth)

    # Get rendered RGB image (shape: H, W, 3)
    render_rgb = gaussian_map.model_map["render_color"]
    if isinstance(render_rgb, torch.Tensor):
        render_rgb = render_rgb.detach().cpu().numpy()
    render_rgb = (np.clip(render_rgb, 0, 1) * 255).astype(np.uint8)
    render_rgb = cv2.cvtColor(render_rgb, cv2.COLOR_RGB2BGR)

    # Get rendered depth image (shape: H, W, 1)
    render_depth = gaussian_map.model_map["render_depth"]
    render_depth_colormap = depth_to_colormap(render_depth, min_depth, max_depth)

    # Get image dimensions
    h, w = camera_rgb.shape[:2]

    # Add labels to images
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

    camera_rgb_labeled = add_label(camera_rgb, "Camera RGB")
    camera_depth_labeled = add_label(camera_depth_colormap, "Camera Depth")
    render_rgb_labeled = add_label(render_rgb, "Rendered RGB")
    render_depth_labeled = add_label(render_depth_colormap, "Rendered Depth")

    # Create 2x2 grid
    top_row = np.hstack([camera_rgb_labeled, camera_depth_labeled])
    bottom_row = np.hstack([render_rgb_labeled, render_depth_labeled])
    grid = np.vstack([top_row, bottom_row])

    return grid


def main():
    global shutdown_requested
    is_debugged = True
    if is_debugged:
        import torch.multiprocessing as mp
        mp.set_start_method('spawn', force=True)  # 或使用 'fork'
    # Register signal handlers for graceful shutdown
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

    # Configure RealSense
    realsense_config = RealsenseConfig(
        width=args_config.realsense_width,
        height=args_config.realsense_height,
        fps=args_config.realsense_fps,
        depth_scale=1000.0,
        align_depth=True,
        enable_auto_exposure=True,
    )

    record_mem = args_config.record_mem

    gaussian_map = Mapping(args_config, time_recorder)
    gaussian_map.create_workspace()
    gaussian_tracker = Tracker(args_config)

    # Save config file
    prepare_cfg(args_config)

    tracker_time_sum = 0
    mapper_time_sum = 0
    frame_id = 0

    # Start RealSense streaming
    print("\n========== Starting RealSense D455 Streaming ==========\n")
    print("Press Ctrl+C to stop...")

    with RealsenseStream(config=realsense_config) as stream:
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

            # Visualization
            if args_config.visualize:
                vis_grid = create_visualization(
                    curr_frame, gaussian_map,
                    min_depth=gaussian_map.min_depth,
                    max_depth=gaussian_map.max_depth
                )
                cv2.imshow("RTG-SLAM Visualization", vis_grid)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == 27:  # 'q' or ESC to quit
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

    if args_config.pcd_densify:
        densify_pcd = gaussian_map.stable_pointcloud.densify(1, 30, 5)
        o3d.io.write_point_cloud(
            os.path.join(args_config.save_path, "save_model", "pcd_densify.ply"), densify_pcd
        )

    print("\n========== SLAM Finished ==========\n")

    # Close visualization window
    if args_config.visualize:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()