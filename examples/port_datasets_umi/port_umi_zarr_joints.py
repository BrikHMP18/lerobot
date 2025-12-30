#!/usr/bin/env python

"""
Convert UMI .zarr dataset to LeRobot format with JOINT ANGLES.

Requires:
    pip install roboticstoolbox-python spatialmath-python

Usage:
    python port_umi_zarr_joints.py \
        --zarr-path ~/dataset.zarr \
        --repo-id username/dataset_joints \
        --robot-type piper
"""

import argparse
import logging
import time
from pathlib import Path

import numpy as np
import zarr
from scipy.spatial.transform import Rotation

# IK dependencies
try:
    import roboticstoolbox as rtb
    from spatialmath import SE3

    HAS_RTB = True
except ImportError:
    HAS_RTB = False
    logging.error(
        "roboticstoolbox-python not installed. Install: pip install roboticstoolbox-python spatialmath-python"
    )

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.utils.utils import get_elapsed_time_in_days_hours_minutes_seconds, init_logging

# Configuration
DEFAULT_UMI_FPS = 30
DEFAULT_ROBOT_TYPE = "piper"

# Zarr paths
ZARR_CAMERA_KEY = "data/camera0_rgb"
ZARR_EEF_POS_KEY = "data/robot0_eef_pos"
ZARR_EEF_ROT_KEY = "data/robot0_eef_rot_axis_angle"
ZARR_GRIPPER_KEY = "data/robot0_gripper_width"
ZARR_EPISODE_ENDS_KEY = "meta/episode_ends"

# Feature schema for JOINT ANGLES
JOINT_FEATURES = {
    "observation.state.joint_positions": {
        "dtype": "float32",
        "shape": (6,),
        "names": {"axes": ["joint0", "joint1", "joint2", "joint3", "joint4", "joint5"]},
    },
    "observation.state.gripper_width": {
        "dtype": "float32",
        "shape": (1,),
        "names": {"axes": ["gripper"]},
    },
    "observation.state": {
        "dtype": "float32",
        "shape": (7,),
        "names": {"axes": ["joint0", "joint1", "joint2", "joint3", "joint4", "joint5", "gripper"]},
    },
    "observation.images.camera0": {
        "dtype": "video",
        "shape": (480, 640, 3),
        "names": ["height", "width", "channels"],
    },
    "action": {
        "dtype": "float32",
        "shape": (7,),
        "names": {"axes": ["joint0", "joint1", "joint2", "joint3", "joint4", "joint5", "gripper"]},
    },
}


def create_robot_model(robot_type: str):
    """Create robot kinematics model for IK."""
    if not HAS_RTB:
        raise ImportError("roboticstoolbox-python required")

    robot_type_lower = robot_type.lower()

    if robot_type_lower == "ur5":
        return rtb.models.UR5()
    elif robot_type_lower == "ur5e":
        return rtb.models.UR5()  # Same kinematics
    elif robot_type_lower == "panda" or robot_type_lower == "franka":
        return rtb.models.Panda()
    elif robot_type_lower == "piper":
        # Load Piper URDF
        urdf_path = Path.home() / "NONHUMAN/piper_urdf/piper_description.urdf"

        if not urdf_path.exists():
            raise FileNotFoundError(
                f"Piper URDF not found at {urdf_path}\n"
                "Download with: mkdir -p ~/NONHUMAN/piper_urdf && "
                "wget -O ~/NONHUMAN/piper_urdf/piper_description.urdf "
                "https://raw.githubusercontent.com/agilexrobotics/piper_ros/noetic/src/piper_description/urdf/piper_description.urdf"
            )

        robot = rtb.Robot.URDF(str(urdf_path))
        logging.info(f"Loaded Piper URDF: {robot.n} links, 6 DOF arm")
        return robot
    else:
        raise ValueError(
            f"Robot '{robot_type}' not supported. "
            f"Supported: ur5, ur5e, panda, franka, piper. "
            f"Add your robot in create_robot_model()"
        )


def tcp_to_joints(eef_pos, eef_rot_axis_angle, robot_model, prev_joints=None):
    """
    Convert TCP pose to joint angles via inverse kinematics.

    Args:
        eef_pos: [x, y, z] in meters
        eef_rot_axis_angle: [rx, ry, rz] in radians
        robot_model: RTB robot model
        prev_joints: Previous joint config for continuity

    Returns:
        joint_angles: [θ1, ..., θ6] in radians
    """
    # Axis-angle to rotation matrix
    rot_matrix = Rotation.from_rotvec(eef_rot_axis_angle).as_matrix()
    tcp_pose = SE3.Rt(rot_matrix, eef_pos)

    # Solve IK with previous solution as initial guess
    q0 = prev_joints if prev_joints is not None else None
    solution = robot_model.ikine_LM(tcp_pose, q0=q0)

    if not solution.success:
        logging.warning(f"IK failed: pos={eef_pos}, rot={eef_rot_axis_angle}")
        return prev_joints if prev_joints is not None else np.zeros(6)

    return solution.q[:6]


def load_zarr_data(zarr_path: Path):
    """Load zarr dataset."""
    if not zarr_path.exists():
        raise FileNotFoundError(f"Zarr not found: {zarr_path}")
    return zarr.open(str(zarr_path), mode="r")


def get_episode_boundaries(zarr_group):
    """Extract episode boundaries."""
    episode_ends = np.array(zarr_group[ZARR_EPISODE_ENDS_KEY])
    boundaries = []
    start = 0
    for end in episode_ends:
        boundaries.append((start, int(end)))
        start = int(end)
    logging.debug(f"Found {len(boundaries)} episodes")
    return boundaries


def auto_detect_image_shape(zarr_group):
    """Detect image dimensions."""
    try:
        first_image = zarr_group[ZARR_CAMERA_KEY][0]
        shape = first_image.shape
        logging.info(f"Detected image shape: {shape}")
        return shape
    except (KeyError, IndexError) as e:
        logging.error(f"Failed to detect image shape: {e}")
        return (480, 640, 3)


def generate_frames_from_episode(zarr_group, start_idx, end_idx, task, robot_model):
    """
    Generate frames with joint angles.

    Converts TCP pose → joints using IK for each frame.
    """
    eef_pos = np.array(zarr_group[ZARR_EEF_POS_KEY][start_idx:end_idx])
    eef_rot = np.array(zarr_group[ZARR_EEF_ROT_KEY][start_idx:end_idx])
    gripper = np.array(zarr_group[ZARR_GRIPPER_KEY][start_idx:end_idx])
    camera = zarr_group[ZARR_CAMERA_KEY][start_idx:end_idx]

    num_frames = end_idx - start_idx
    prev_joints = None
    ik_failures = 0

    for i in range(num_frames):
        # Convert observation TCP → joints
        obs_joints = tcp_to_joints(eef_pos[i], eef_rot[i], robot_model, prev_joints)
        if prev_joints is not None and np.allclose(obs_joints, prev_joints):
            ik_failures += 1

        obs_gripper = gripper[i : i + 1]
        obs_state = np.concatenate([obs_joints, obs_gripper])

        # Convert action TCP → joints
        if i < num_frames - 1:
            action_joints = tcp_to_joints(eef_pos[i + 1], eef_rot[i + 1], robot_model, obs_joints)
            action_gripper = gripper[i + 1 : i + 2]
        else:
            action_joints = obs_joints
            action_gripper = obs_gripper

        action = np.concatenate([action_joints, action_gripper])
        prev_joints = obs_joints

        frame = {
            "observation.images.camera0": np.array(camera[i]),
            "observation.state": obs_state.astype(np.float32),
            "observation.state.joint_positions": obs_joints.astype(np.float32),
            "observation.state.gripper_width": obs_gripper.astype(np.float32),
            "action": action.astype(np.float32),
            "task": task,
        }

        yield frame

    if ik_failures > 0:
        logging.warning(f"IK failures in episode: {ik_failures}/{num_frames} frames")


def port_umi_zarr(
    zarr_path: Path,
    repo_id: str,
    output_dir: Path | None = None,
    fps: int = DEFAULT_UMI_FPS,
    robot_type: str = DEFAULT_ROBOT_TYPE,
    task_description: str = "manipulation",
    push_to_hub: bool = False,
):
    """Convert UMI .zarr to LeRobot with joint angles."""
    start_time = time.time()

    # Create robot model
    logging.info(f"Creating robot model: {robot_type}")
    robot_model = create_robot_model(robot_type)

    # Load zarr
    zarr_group = load_zarr_data(zarr_path)
    episode_boundaries = get_episode_boundaries(zarr_group)
    num_episodes = len(episode_boundaries)

    # Detect image shape
    image_shape = auto_detect_image_shape(zarr_group)
    JOINT_FEATURES["observation.images.camera0"]["shape"] = list(image_shape)

    # Create dataset
    logging.info(f"Creating dataset: {repo_id}")
    lerobot_dataset = LeRobotDataset.create(
        repo_id=repo_id,
        root=output_dir,
        robot_type=robot_type,
        fps=fps,
        features=JOINT_FEATURES,
        use_videos=True,
    )

    # Process episodes
    for ep_idx, (start_idx, end_idx) in enumerate(episode_boundaries):
        elapsed = time.time() - start_time
        d, h, m, s = get_elapsed_time_in_days_hours_minutes_seconds(elapsed)

        logging.info(
            f"Episode {ep_idx + 1}/{num_episodes} (frames {start_idx}-{end_idx}) [{d}d {h}h {m}m {s:.1f}s]"
        )

        for frame in generate_frames_from_episode(
            zarr_group, start_idx, end_idx, task_description, robot_model
        ):
            lerobot_dataset.add_frame(frame)

        lerobot_dataset.save_episode()

    # Finalize
    lerobot_dataset.finalize(push_to_hub=push_to_hub)

    elapsed = time.time() - start_time
    d, h, m, s = get_elapsed_time_in_days_hours_minutes_seconds(elapsed)
    logging.info(f"✅ Conversion complete in {d}d {h}h {m}m {s:.1f}s")

    return lerobot_dataset


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Convert UMI .zarr to LeRobot with joint angles",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("--zarr-path", type=Path, required=True, help="Path to dataset.zarr")
    parser.add_argument("--repo-id", type=str, required=True, help="HuggingFace repo ID")
    parser.add_argument("--output-dir", type=Path, default=None, help="Local output directory")
    parser.add_argument("--fps", type=int, default=DEFAULT_UMI_FPS, help="Frames per second")
    parser.add_argument(
        "--robot-type", type=str, default=DEFAULT_ROBOT_TYPE, help="Robot type (ur5, panda, etc)"
    )
    parser.add_argument("--task-description", type=str, default="manipulation", help="Task description")
    parser.add_argument("--push-to-hub", action="store_true", help="Push to HuggingFace Hub")

    args = parser.parse_args()
    init_logging()

    port_umi_zarr(
        zarr_path=args.zarr_path,
        repo_id=args.repo_id,
        output_dir=args.output_dir,
        fps=args.fps,
        robot_type=args.robot_type,
        task_description=args.task_description,
        push_to_hub=args.push_to_hub,
    )


if __name__ == "__main__":
    main()
