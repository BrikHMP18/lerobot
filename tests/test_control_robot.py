#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from unittest.mock import patch

from lerobot.scripts.lerobot_calibrate import CalibrateConfig, calibrate
from lerobot.scripts.lerobot_record import DatasetRecordConfig, RecordConfig, record
from lerobot.scripts.lerobot_replay import DatasetReplayConfig, ReplayConfig, replay
from lerobot.scripts.lerobot_teleoperate import TeleoperateConfig, teleoperate
from tests.fixtures.constants import DUMMY_REPO_ID
from tests.mocks.mock_robot import MockRobotConfig
from tests.mocks.mock_teleop import MockTeleopConfig


def test_calibrate():
    robot_cfg = MockRobotConfig()
    cfg = CalibrateConfig(robot=robot_cfg)
    calibrate(cfg)


def test_teleoperate():
    robot_cfg = MockRobotConfig()
    teleop_cfg = MockTeleopConfig()
    cfg = TeleoperateConfig(
        robot=robot_cfg,
        teleop=teleop_cfg,
        teleop_time_s=0.1,
    )
    teleoperate(cfg)


def test_record_and_resume(tmp_path):
    robot_cfg = MockRobotConfig()
    teleop_cfg = MockTeleopConfig()
    dataset_cfg = DatasetRecordConfig(
        repo_id=DUMMY_REPO_ID,
        single_task="Dummy task",
        root=tmp_path / "record",
        num_episodes=1,
        episode_time_s=0.1,
        reset_time_s=0,
        push_to_hub=False,
    )
    cfg = RecordConfig(
        robot=robot_cfg,
        dataset=dataset_cfg,
        teleop=teleop_cfg,
        play_sounds=False,
    )

    dataset = record(cfg)

    assert dataset.fps == 30
    assert dataset.meta.total_episodes == dataset.num_episodes == 1
    assert dataset.meta.total_frames == dataset.num_frames == 3
    assert dataset.meta.total_tasks == 1

    cfg.resume = True
    # Mock the revision to prevent Hub calls during resume
    with (
        patch("lerobot.datasets.lerobot_dataset.get_safe_version") as mock_get_safe_version,
        patch("lerobot.datasets.lerobot_dataset.snapshot_download") as mock_snapshot_download,
    ):
        mock_get_safe_version.return_value = "v3.0"
        mock_snapshot_download.return_value = str(tmp_path / "record")
        dataset = record(cfg)

    assert dataset.meta.total_episodes == dataset.num_episodes == 2
    assert dataset.meta.total_frames == dataset.num_frames == 6
    assert dataset.meta.total_tasks == 1


def test_record_and_replay(tmp_path):
    robot_cfg = MockRobotConfig()
    teleop_cfg = MockTeleopConfig()
    record_dataset_cfg = DatasetRecordConfig(
        repo_id=DUMMY_REPO_ID,
        single_task="Dummy task",
        root=tmp_path / "record_and_replay",
        num_episodes=1,
        episode_time_s=0.1,
        push_to_hub=False,
    )
    record_cfg = RecordConfig(
        robot=robot_cfg,
        dataset=record_dataset_cfg,
        teleop=teleop_cfg,
        play_sounds=False,
    )
    replay_dataset_cfg = DatasetReplayConfig(
        repo_id=DUMMY_REPO_ID,
        episode=0,
        root=tmp_path / "record_and_replay",
    )
    replay_cfg = ReplayConfig(
        robot=robot_cfg,
        dataset=replay_dataset_cfg,
        play_sounds=False,
    )

    record(record_cfg)

    # Mock the revision to prevent Hub calls during replay
    with (
        patch("lerobot.datasets.lerobot_dataset.get_safe_version") as mock_get_safe_version,
        patch("lerobot.datasets.lerobot_dataset.snapshot_download") as mock_snapshot_download,
    ):
        mock_get_safe_version.return_value = "v3.0"
        mock_snapshot_download.return_value = str(tmp_path / "record_and_replay")
        replay(replay_cfg)


def test_record_manual_right_then_esc_saves_two_episodes(tmp_path):
    robot_cfg = MockRobotConfig()
    teleop_cfg = MockTeleopConfig()
    dataset_cfg = DatasetRecordConfig(
        repo_id=DUMMY_REPO_ID,
        single_task="Dummy task",
        root=tmp_path / "manual_right_then_esc",
        num_episodes=1,
        push_to_hub=False,
        manual_episode_control=True,
    )
    cfg = RecordConfig(
        robot=robot_cfg,
        dataset=dataset_cfg,
        teleop=teleop_cfg,
        play_sounds=False,
    )

    events = {"exit_early": False, "rerecord_episode": False, "stop_recording": False}
    save_calls = []
    step = {"value": 0}

    def dummy_record_loop(*args, **kwargs):
        dataset = kwargs["dataset"]
        dataset.episode_buffer["size"] = 1

        if step["value"] == 0:
            events["exit_early"] = True
        else:
            events["stop_recording"] = True
            events["exit_early"] = True
        step["value"] += 1

    def dummy_save_episode(self, *args, **kwargs):
        save_calls.append(1)
        self.episode_buffer = self.create_episode_buffer()

    with (
        patch("lerobot.scripts.lerobot_record.init_keyboard_listener", return_value=(None, events)),
        patch("lerobot.scripts.lerobot_record.record_loop", side_effect=dummy_record_loop),
        patch("lerobot.datasets.lerobot_dataset.LeRobotDataset.save_episode", new=dummy_save_episode),
    ):
        record(cfg)

    assert len(save_calls) == 2


def test_record_manual_left_rerecord_discards_episode(tmp_path):
    robot_cfg = MockRobotConfig()
    teleop_cfg = MockTeleopConfig()
    dataset_cfg = DatasetRecordConfig(
        repo_id=DUMMY_REPO_ID,
        single_task="Dummy task",
        root=tmp_path / "manual_left_rerecord",
        num_episodes=1,
        push_to_hub=False,
        manual_episode_control=True,
    )
    cfg = RecordConfig(
        robot=robot_cfg,
        dataset=dataset_cfg,
        teleop=teleop_cfg,
        play_sounds=False,
    )

    events = {"exit_early": False, "rerecord_episode": False, "stop_recording": False}
    save_calls = []
    clear_calls = []
    step = {"value": 0}

    def dummy_record_loop(*args, **kwargs):
        dataset = kwargs["dataset"]
        dataset.episode_buffer["size"] = 1

        if step["value"] == 0:
            events["rerecord_episode"] = True
            events["exit_early"] = True
        elif step["value"] == 1:
            events["exit_early"] = True
        else:
            events["stop_recording"] = True
            events["exit_early"] = True
        step["value"] += 1

    def dummy_save_episode(self, *args, **kwargs):
        save_calls.append(1)
        self.episode_buffer = self.create_episode_buffer()

    def dummy_clear_episode_buffer(self, *args, **kwargs):
        clear_calls.append(1)
        self.episode_buffer = self.create_episode_buffer()

    with (
        patch("lerobot.scripts.lerobot_record.init_keyboard_listener", return_value=(None, events)),
        patch("lerobot.scripts.lerobot_record.record_loop", side_effect=dummy_record_loop),
        patch("lerobot.datasets.lerobot_dataset.LeRobotDataset.save_episode", new=dummy_save_episode),
        patch(
            "lerobot.datasets.lerobot_dataset.LeRobotDataset.clear_episode_buffer",
            new=dummy_clear_episode_buffer,
        ),
    ):
        record(cfg)

    assert len(clear_calls) == 1
    assert len(save_calls) == 2


def test_record_manual_esc_saves_partial_episode(tmp_path):
    robot_cfg = MockRobotConfig()
    teleop_cfg = MockTeleopConfig()
    dataset_cfg = DatasetRecordConfig(
        repo_id=DUMMY_REPO_ID,
        single_task="Dummy task",
        root=tmp_path / "manual_esc_partial",
        num_episodes=1,
        push_to_hub=False,
        manual_episode_control=True,
    )
    cfg = RecordConfig(
        robot=robot_cfg,
        dataset=dataset_cfg,
        teleop=teleop_cfg,
        play_sounds=False,
    )

    events = {"exit_early": False, "rerecord_episode": False, "stop_recording": False}
    save_calls = []

    def dummy_record_loop(*args, **kwargs):
        dataset = kwargs["dataset"]
        dataset.episode_buffer["size"] = 1
        events["stop_recording"] = True
        events["exit_early"] = True

    def dummy_save_episode(self, *args, **kwargs):
        save_calls.append(1)
        self.episode_buffer = self.create_episode_buffer()

    with (
        patch("lerobot.scripts.lerobot_record.init_keyboard_listener", return_value=(None, events)),
        patch("lerobot.scripts.lerobot_record.record_loop", side_effect=dummy_record_loop),
        patch("lerobot.datasets.lerobot_dataset.LeRobotDataset.save_episode", new=dummy_save_episode),
    ):
        record(cfg)

    assert len(save_calls) == 1
