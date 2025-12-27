#!/usr/bin/env python

"""
Convert Universal Manipulation Interface (UMI) .zarr dataset to LeRobot format.

Usage:
    python port_umi_zarr.py \
        --zarr-path ~/path/to/dataset.zarr \
        --repo-id username/umi_dataset \
        --fps 30
"""

import argparse
import logging
import time
from pathlib import Path

import numpy as np
import zarr

from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.utils.utils import get_elapsed_time_in_days_hours_minutes_seconds, init_logging

# UMI Dataset Configuration
DEFAULT_UMI_FPS = 30  # Default FPS for UMI recordings
DEFAULT_ROBOT_TYPE = "UMI"

# Zarr data paths
ZARR_CAMERA_KEY = "data/camera0_rgb"
ZARR_EEF_POS_KEY = "data/robot0_eef_pos"
ZARR_EEF_ROT_KEY = "data/robot0_eef_rot_axis_angle"
ZARR_GRIPPER_KEY = "data/robot0_gripper_width"
ZARR_EPISODE_ENDS_KEY = "meta/episode_ends"

# LeRobot feature schema for UMI dataset
UMI_FEATURES = {
    # Observation state components
    "observation.state.eef_pos": {
        "dtype": "float32",
        "shape": (3,),
        "names": {
            "axes": ["x", "y", "z"],
        },
    },
    "observation.state.eef_rot_axis_angle": {
        "dtype": "float32",
        "shape": (3,),
        "names": {
            "axes": ["rx", "ry", "rz"],
        },
    },
    "observation.state.gripper_width": {
        "dtype": "float32",
        "shape": (1,),
        "names": {
            "axes": ["gripper"],
        },
    },
    # Combined state (standard LeRobot format)
    "observation.state": {
        "dtype": "float32",
        "shape": (7,),
        "names": {
            "axes": ["x", "y", "z", "rx", "ry", "rz", "gripper"],
        },
    },
    # Camera observations (will be converted to video)
    "observation.images.camera0": {
        "dtype": "video",
        "shape": (480, 640, 3),  # Will be auto-detected from actual data
        "names": ["height", "width", "channels"],
    },
    # Actions (next state targets)
    "action": {
        "dtype": "float32",
        "shape": (7,),
        "names": {
            "axes": ["x", "y", "z", "rx", "ry", "rz", "gripper"],
        },
    },
}


def load_zarr_data(zarr_path: Path):
    """Load zarr dataset and return zarr group"""
    logging.info(f"Loading zarr from: {zarr_path}")
    return zarr.open(str(zarr_path), mode="r")


def get_episode_boundaries(zarr_group):
    """
    Extract episode start/end indices from meta/episode_ends
    Returns list of (start_idx, end_idx) tuples
    """
    episode_ends = np.array(zarr_group[ZARR_EPISODE_ENDS_KEY])
    episode_starts = np.concatenate([[0], episode_ends[:-1]])

    episodes = list(zip(episode_starts, episode_ends, strict=True))
    logging.info(f"Found {len(episodes)} episodes")
    logging.debug(f"Episode boundaries: {episodes}")

    return episodes


def auto_detect_image_shape(zarr_group):
    """Auto-detect image shape from the first frame"""
    try:
        first_frame = np.array(zarr_group[ZARR_CAMERA_KEY][0])
        shape = tuple(first_frame.shape)
        logging.info(f"Auto-detected image shape: {shape}")
        return shape
    except (KeyError, IndexError) as e:
        logging.warning(f"Could not auto-detect image shape: {e}. Using default (480, 640, 3)")
        return (480, 640, 3)


def generate_frames_from_episode(
    zarr_group,
    start_idx: int,
    end_idx: int,
    task_description: str = "umi_manipulation_task",
):
    """
    Generator that yields frames in LeRobot format from a single episode

    Args:
        zarr_group: Open zarr group
        start_idx: Starting frame index
        end_idx: Ending frame index
        task_description: Task name/description

    Yields:
        dict: Frame data in LeRobot format
    """
    episode_length = end_idx - start_idx
    logging.debug(f"Processing episode from {start_idx} to {end_idx} ({episode_length} frames)")

    # Load all episode data at once for efficiency
    eef_pos = np.array(zarr_group[ZARR_EEF_POS_KEY][start_idx:end_idx])
    eef_rot = np.array(zarr_group[ZARR_EEF_ROT_KEY][start_idx:end_idx])
    gripper_width = np.array(zarr_group[ZARR_GRIPPER_KEY][start_idx:end_idx])
    camera_rgb = np.array(zarr_group[ZARR_CAMERA_KEY][start_idx:end_idx])

    # Generate frames
    for frame_idx in range(episode_length):
        # Current observations
        obs_eef_pos = eef_pos[frame_idx].astype(np.float32)
        obs_eef_rot = eef_rot[frame_idx].astype(np.float32)
        obs_gripper = gripper_width[frame_idx].reshape(1).astype(np.float32)
        obs_image = camera_rgb[frame_idx]  # (H, W, C) uint8

        # Action = next state (or current state if last frame)
        if frame_idx < episode_length - 1:
            action_eef_pos = eef_pos[frame_idx + 1].astype(np.float32)
            action_eef_rot = eef_rot[frame_idx + 1].astype(np.float32)
            action_gripper = gripper_width[frame_idx + 1].reshape(1).astype(np.float32)
        else:
            action_eef_pos = obs_eef_pos
            action_eef_rot = obs_eef_rot
            action_gripper = obs_gripper

        # Combine into full state and action vectors
        obs_state = np.concatenate([obs_eef_pos, obs_eef_rot, obs_gripper])
        action = np.concatenate([action_eef_pos, action_eef_rot, action_gripper])

        # Create frame dictionary
        frame = {
            # Individual observations
            "observation.state.eef_pos": obs_eef_pos,
            "observation.state.eef_rot_axis_angle": obs_eef_rot,
            "observation.state.gripper_width": obs_gripper,
            # Combined state
            "observation.state": obs_state,
            # Image
            "observation.images.camera0": obs_image,
            # Action
            "action": action,
            # Task (handled separately by add_frame)
            "task": task_description,
        }

        yield frame


def port_umi_zarr(
    zarr_path: Path,
    repo_id: str,
    output_dir: Path | None = None,
    fps: int = DEFAULT_UMI_FPS,
    robot_type: str = DEFAULT_ROBOT_TYPE,
    task_description: str = "umi_manipulation",
    push_to_hub: bool = False,
):
    """
    Convert UMI .zarr dataset to LeRobot format

    Args:
        zarr_path: Path to dataset.zarr directory
        repo_id: Repository ID (e.g., "username/dataset-name")
        output_dir: Output directory (default: ~/.cache/huggingface/lerobot)
        fps: Frames per second
        robot_type: Robot type identifier
        task_description: Task description string
        push_to_hub: Whether to push to HuggingFace Hub

    Returns:
        LeRobotDataset: Converted dataset
    """
    start_time = time.time()

    # Load zarr data
    zarr_group = load_zarr_data(zarr_path)

    # Get episode boundaries
    episode_boundaries = get_episode_boundaries(zarr_group)
    num_episodes = len(episode_boundaries)

    # Auto-detect image shape
    image_shape = auto_detect_image_shape(zarr_group)
    UMI_FEATURES["observation.images.camera0"]["shape"] = list(image_shape)

    # Create LeRobot dataset
    logging.info(f"Creating LeRobot dataset: {repo_id}")
    lerobot_dataset = LeRobotDataset.create(
        repo_id=repo_id,
        root=output_dir,
        robot_type=robot_type,
        fps=fps,
        features=UMI_FEATURES,
        use_videos=True,  # Store images as videos
    )

    # Process each episode
    for ep_idx, (start_idx, end_idx) in enumerate(episode_boundaries):
        elapsed = time.time() - start_time
        d, h, m, s = get_elapsed_time_in_days_hours_minutes_seconds(elapsed)

        logging.info(
            f"Episode {ep_idx + 1}/{num_episodes} "
            f"(frames {start_idx}-{end_idx}) "
            f"[{d}d {h}h {m}m {s:.1f}s elapsed]"
        )

        # Generate and add frames
        for frame in generate_frames_from_episode(zarr_group, start_idx, end_idx, task_description):
            lerobot_dataset.add_frame(frame)

        # Save episode
        lerobot_dataset.save_episode()
        logging.info(f"✓ Episode {ep_idx + 1} saved")

    # Finalize dataset
    logging.info("Finalizing dataset...")
    lerobot_dataset.finalize()

    elapsed = time.time() - start_time
    d, h, m, s = get_elapsed_time_in_days_hours_minutes_seconds(elapsed)
    logging.info(f"✅ Dataset conversion complete! [{d}d {h}h {m}m {s:.1f}s total]")
    logging.info(f"📁 Dataset location: {lerobot_dataset.root}")

    # Push to hub if requested
    if push_to_hub:
        logging.info("Pushing to HuggingFace Hub...")
        lerobot_dataset.push_to_hub(
            tags=["robotics", "manipulation", "umi", "imitation-learning", "6dof"],
            private=False,
        )
        logging.info(f"✅ Dataset pushed to: https://huggingface.co/datasets/{repo_id}")

    return lerobot_dataset


def validate_dataset(repo_id: str):
    """Sanity check that ensures metadata can be loaded and all files are present."""
    meta = LeRobotDatasetMetadata(repo_id)

    if meta.total_episodes == 0:
        raise ValueError("Number of episodes is 0.")

    for ep_idx in range(meta.total_episodes):
        data_path = meta.root / meta.get_data_file_path(ep_idx)

        if not data_path.exists():
            raise ValueError(f"Parquet file is missing in: {data_path}")

        for vid_key in meta.video_keys:
            vid_path = meta.root / meta.get_video_file_path(ep_idx, vid_key)
            if not vid_path.exists():
                raise ValueError(f"Video file is missing in: {vid_path}")

    logging.info(f"✅ Dataset validation passed for {repo_id}")


def main():
    """
    Main entry point for converting UMI .zarr datasets to LeRobot format.

    Parses command-line arguments and executes the conversion pipeline,
    optionally pushing the result to HuggingFace Hub.
    """
    parser = argparse.ArgumentParser(
        description="Convert UMI .zarr dataset to LeRobot format",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--zarr-path",
        type=Path,
        required=True,
        help="Path to dataset.zarr directory",
    )
    parser.add_argument(
        "--repo-id",
        type=str,
        required=True,
        help="Repository ID (e.g., 'username/dataset-name')",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory (default: ~/.cache/huggingface/lerobot)",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=DEFAULT_UMI_FPS,
        help=f"Frames per second (default: {DEFAULT_UMI_FPS})",
    )
    parser.add_argument(
        "--robot-type",
        type=str,
        default=DEFAULT_ROBOT_TYPE,
        help=f"Robot type identifier (default: {DEFAULT_ROBOT_TYPE})",
    )
    parser.add_argument(
        "--task-description",
        type=str,
        default="umi_manipulation",
        help="Task description (default: 'umi_manipulation')",
    )
    parser.add_argument(
        "--push-to-hub",
        action="store_true",
        help="Push dataset to HuggingFace Hub after conversion",
    )

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
