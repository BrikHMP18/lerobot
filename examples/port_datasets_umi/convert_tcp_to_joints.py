#!/usr/bin/env python

"""Convert LeRobot TCP pose dataset to joint angles using Piper DH parameters.

Converts TCP pose format [x, y, z, rx, ry, rz, gripper] to joint angles [joint0...joint5, gripper]
using numerical inverse kinematics with official AgileX Piper DH parameters.

Usage:
    python convert_tcp_to_joints.py \
        --input-dataset path/to/tcp_dataset \
        --output-repo-id org/dataset_joints \
        --push-to-hub
"""

import argparse
import logging
import time
from pathlib import Path

import numpy as np
from piper_kinematics import PiperInverseKinematics

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.utils.utils import get_elapsed_time_in_days_hours_minutes_seconds, init_logging


def convert_tcp_to_joints(
    input_dataset_path: Path,
    output_repo_id: str,
    output_dir: Path | None = None,
    dh_is_offset: int = 0x01,
    push_to_hub: bool = False,
):
    """Convert TCP pose dataset to joint angles format.

    Args:
        input_dataset_path: Path to TCP format dataset
        output_repo_id: Output repository ID (e.g., 'org/dataset_joints')
        output_dir: Output directory (default: ~/.cache/huggingface/lerobot)
        dh_is_offset: DH version (0x00=old, 0x01=new firmware)
        push_to_hub: Push to HuggingFace Hub after conversion
    """
    start_time = time.time()

    logging.info(f"Loading TCP dataset: {input_dataset_path}")
    tcp_dataset = LeRobotDataset(input_dataset_path, video_backend="pyav")

    ik_solver = PiperInverseKinematics(dh_is_offset)

    num_episodes = tcp_dataset.num_episodes
    num_frames = len(tcp_dataset)
    fps = tcp_dataset.fps

    logging.info(f"Dataset: {num_episodes} episodes, {num_frames} frames @ {fps} fps")

    joint_features = {
        "observation.images.camera0": tcp_dataset.features["observation.images.camera0"],
        "observation.state": {
            "dtype": "float32",
            "shape": (7,),
            "names": {"axes": ["joint0", "joint1", "joint2", "joint3", "joint4", "joint5", "gripper"]},
        },
        "action": {
            "dtype": "float32",
            "shape": (7,),
            "names": {"axes": ["joint0", "joint1", "joint2", "joint3", "joint4", "joint5", "gripper"]},
        },
    }

    logging.info(f"Creating output dataset: {output_repo_id}")
    joint_dataset = LeRobotDataset.create(
        repo_id=output_repo_id,
        root=output_dir,
        robot_type="piper",
        fps=fps,
        features=joint_features,
        use_videos=True,
    )

    ik_failures = 0
    ik_total = 0
    previous_joints = np.zeros(6)

    for ep_idx in range(num_episodes):
        elapsed = time.time() - start_time
        d, h, m, s = get_elapsed_time_in_days_hours_minutes_seconds(elapsed)
        logging.info(f"Episode {ep_idx + 1}/{num_episodes} (after {d}d {h}h {m}m {s:.1f}s)")

        episode_meta = tcp_dataset.meta.episodes[ep_idx]
        ep_start = episode_meta["dataset_from_index"]
        ep_end = episode_meta["dataset_to_index"]

        for frame_idx in range(ep_end - ep_start):
            frame = tcp_dataset[ep_start + frame_idx]

            # Extract TCP poses
            obs_tcp = frame["observation.state"][:6].numpy()
            obs_gripper = frame["observation.state"][6].item()
            action_tcp = frame["action"][:6].numpy()
            action_gripper = frame["action"][6].item()

            # Solve IK
            obs_joints, obs_success, obs_error = ik_solver.solve_ik(obs_tcp, initial_joints=previous_joints)
            action_joints, action_success, _ = ik_solver.solve_ik(action_tcp, initial_joints=obs_joints)

            ik_total += 2
            if not obs_success:
                ik_failures += 1
                logging.warning(f"IK failed: ep{ep_idx} frame{frame_idx} obs (error={obs_error:.1f}mm)")
            if not action_success:
                ik_failures += 1

            previous_joints = obs_joints

            # Prepare frame
            task = tcp_dataset.meta.tasks.iloc[frame["task_index"].item()].name
            image = frame["observation.images.camera0"]
            if image.shape[0] == 3:
                image = image.permute(1, 2, 0)

            joint_dataset.add_frame(
                {
                    "observation.images.camera0": image,
                    "observation.state": np.concatenate([obs_joints, [obs_gripper]]).astype(np.float32),
                    "action": np.concatenate([action_joints, [action_gripper]]).astype(np.float32),
                    "task": task,
                }
            )

        joint_dataset.save_episode()

    joint_dataset.finalize()

    if push_to_hub:
        logging.info(f"Pushing to Hub: {output_repo_id}")
        joint_dataset.push_to_hub()

    elapsed = time.time() - start_time
    d, h, m, s = get_elapsed_time_in_days_hours_minutes_seconds(elapsed)
    success_rate = (1 - ik_failures / ik_total) * 100 if ik_total > 0 else 0

    logging.info(f"Conversion complete in {d}d {h}h {m}m {s:.1f}s")
    logging.info(f"IK success: {success_rate:.1f}% ({ik_total - ik_failures}/{ik_total})")
    if ik_failures > 0:
        logging.warning(f"{ik_failures} IK failures - poses may be unreachable")

    return joint_dataset


def main():
    parser = argparse.ArgumentParser(description="Convert TCP pose dataset to joint angles")
    parser.add_argument("--input-dataset", type=Path, required=True, help="Path to TCP dataset")
    parser.add_argument(
        "--output-repo-id", type=str, required=True, help="Output repo (e.g., 'org/dataset_joints')"
    )
    parser.add_argument("--output-dir", type=Path, default=None, help="Output directory")
    parser.add_argument(
        "--dh-is-offset",
        type=int,
        default=0x01,
        choices=[0x00, 0x01],
        help="DH version: 0x00=old firmware, 0x01=new (>=S-V1.6-3)",
    )
    parser.add_argument("--push-to-hub", action="store_true", help="Push to HuggingFace Hub")

    args = parser.parse_args()
    init_logging()

    convert_tcp_to_joints(
        input_dataset_path=args.input_dataset,
        output_repo_id=args.output_repo_id,
        output_dir=args.output_dir,
        dh_is_offset=args.dh_is_offset,
        push_to_hub=args.push_to_hub,
    )


if __name__ == "__main__":
    main()
