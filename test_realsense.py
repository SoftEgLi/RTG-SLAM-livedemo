"""
Test script to diagnose RealSense D455 depth issues
"""
import numpy as np
import time

try:
    import pyrealsense2 as rs
except ImportError:
    raise ImportError("Please install pyrealsense2: pip install pyrealsense2")


def test_realsense():
    print("=" * 60)
    print("RealSense D455 Diagnostic Test")
    print("=" * 60)

    # Create pipeline
    pipeline = rs.pipeline()
    config = rs.config()

    # Get device info
    ctx = rs.context()
    devices = ctx.query_devices()
    if len(devices) == 0:
        print("[ERROR] No RealSense device found!")
        return

    dev = devices[0]
    print(f"\n[Device Info]")
    print(f"  Name: {dev.get_info(rs.camera_info.name)}")
    print(f"  Serial: {dev.get_info(rs.camera_info.serial_number)}")
    print(f"  Firmware: {dev.get_info(rs.camera_info.firmware_version)}")

    # List available streams
    print(f"\n[Available Streams]")
    for sensor in dev.sensors:
        print(f"  Sensor: {sensor.get_info(rs.camera_info.name)}")
        for profile in sensor.get_stream_profiles():
            if isinstance(profile, rs.video_stream_profile):
                print(f"    - {profile.stream_type()} {profile.format()} "
                      f"{profile.width()}x{profile.height()} @ {profile.fps()}fps")

    # Try different configurations
    test_configs = [
        (640, 480, 30),
        (848, 480, 30),
        (1280, 720, 15),
    ]

    for width, height, fps in test_configs:
        print(f"\n[Testing {width}x{height} @ {fps}fps]")
        try:
            config = rs.config()
            config.enable_stream(rs.stream.color, width, height, rs.format.rgb8, fps)
            config.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)

            profile = pipeline.start(config)

            # Get depth scale
            depth_sensor = profile.get_device().first_depth_sensor()
            depth_scale = depth_sensor.get_depth_scale()
            print(f"  Depth scale: {depth_scale} (1 unit = {depth_scale*1000:.3f} mm)")

            # Get depth intrinsics
            depth_stream = profile.get_stream(rs.stream.depth)
            depth_intrinsics = depth_stream.as_video_stream_profile().get_intrinsics()
            print(f"  Depth intrinsics: {depth_intrinsics.width}x{depth_intrinsics.height}")
            print(f"    fx={depth_intrinsics.fx:.2f}, fy={depth_intrinsics.fy:.2f}")
            print(f"    cx={depth_intrinsics.ppx:.2f}, cy={depth_intrinsics.ppy:.2f}")

            # Create align object
            align = rs.align(rs.stream.color)

            # Skip some frames to let auto-exposure stabilize
            print("  Warming up (skipping 30 frames)...")
            for _ in range(30):
                pipeline.wait_for_frames()

            # Capture test frames
            print("  Capturing test frames...")
            for i in range(5):
                frames = pipeline.wait_for_frames()

                # Test without alignment
                depth_frame_raw = frames.get_depth_frame()
                color_frame = frames.get_color_frame()

                if depth_frame_raw:
                    depth_raw = np.asanyarray(depth_frame_raw.get_data())
                    print(f"\n  Frame {i+1} (raw depth):")
                    print(f"    Shape: {depth_raw.shape}, dtype: {depth_raw.dtype}")
                    print(f"    Min: {depth_raw.min()}, Max: {depth_raw.max()}")
                    print(f"    Non-zero pixels: {np.count_nonzero(depth_raw)} / {depth_raw.size} "
                          f"({100*np.count_nonzero(depth_raw)/depth_raw.size:.1f}%)")
                    if depth_raw.max() > 0:
                        print(f"    Depth range (meters): {depth_raw.min()*depth_scale:.3f} - {depth_raw.max()*depth_scale:.3f}")

                # Test with alignment
                aligned_frames = align.process(frames)
                depth_frame_aligned = aligned_frames.get_depth_frame()

                if depth_frame_aligned:
                    depth_aligned = np.asanyarray(depth_frame_aligned.get_data())
                    print(f"  Frame {i+1} (aligned depth):")
                    print(f"    Shape: {depth_aligned.shape}, dtype: {depth_aligned.dtype}")
                    print(f"    Min: {depth_aligned.min()}, Max: {depth_aligned.max()}")
                    print(f"    Non-zero pixels: {np.count_nonzero(depth_aligned)} / {depth_aligned.size} "
                          f"({100*np.count_nonzero(depth_aligned)/depth_aligned.size:.1f}%)")

                if color_frame:
                    color = np.asanyarray(color_frame.get_data())
                    print(f"  Frame {i+1} (color):")
                    print(f"    Shape: {color.shape}, dtype: {color.dtype}")
                    print(f"    Min: {color.min()}, Max: {color.max()}")

                time.sleep(0.1)

            pipeline.stop()
            print(f"  [OK] Configuration works!")

        except Exception as e:
            print(f"  [FAILED] {e}")
            try:
                pipeline.stop()
            except:
                pass

    print("\n" + "=" * 60)
    print("Test complete!")
    print("=" * 60)


if __name__ == "__main__":
    test_realsense()
