#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
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

from types import SimpleNamespace

from lerobot.rl.wandb_utils import WandBLogger


def test_wandb_logger_accepts_val_mode_for_log_dict():
    logger = WandBLogger.__new__(WandBLogger)
    logger._wandb_custom_step_key = None
    logger._wandb = SimpleNamespace(
        log=lambda *args, **kwargs: None,
        define_metric=lambda *args, **kwargs: None,
    )

    # Should not raise.
    logger.log_dict({"loss": 1.23}, step=10, mode="val")


def test_wandb_logger_accepts_val_mode_for_video():
    logger = WandBLogger.__new__(WandBLogger)
    logger.env_fps = 30

    class DummyWandb:
        @staticmethod
        def Video(path, fps, format):
            return {"path": path, "fps": fps, "format": format}

        @staticmethod
        def log(*args, **kwargs):
            return None

    logger._wandb = DummyWandb()

    # Should not raise.
    logger.log_video("dummy.mp4", step=10, mode="val")
