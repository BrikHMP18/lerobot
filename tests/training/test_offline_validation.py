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

from contextlib import nullcontext

import torch

from lerobot.scripts.lerobot_train import flatten_output_metrics, run_offline_validation


class DummyAccelerator:
    def unwrap_model(self, policy, keep_fp32_wrapper=True):
        return policy

    def autocast(self):
        return nullcontext()


class DummyPolicy(torch.nn.Module):
    def forward(self, batch):
        # Mean over batch as scalar loss.
        loss = batch["loss_value"].mean()
        # Mix scalar and vector metrics to exercise flattening.
        output_dict = {"scalar_metric": 3.0, "vector_metric": [1.0, 2.0]}
        return loss, output_dict


def test_flatten_output_metrics_expands_non_scalars():
    output = {
        "scalar": 1.5,
        "tensor_vec": torch.tensor([2.0, 3.0]),
        "list_vec": [4.0, 5.0],
        "text": "ok",
    }

    flat = flatten_output_metrics(output)

    assert flat["scalar"] == 1.5
    assert flat["tensor_vec_0"] == 2.0
    assert flat["tensor_vec_1"] == 3.0
    assert flat["list_vec_0"] == 4.0
    assert flat["list_vec_1"] == 5.0
    assert flat["text"] == "ok"


def test_run_offline_validation_averages_loss_and_metrics():
    dataset = [
        {"loss_value": torch.tensor([1.0], dtype=torch.float32)},
        {"loss_value": torch.tensor([3.0], dtype=torch.float32)},
    ]
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)

    metrics = run_offline_validation(
        policy=DummyPolicy(),
        val_dataloader=dataloader,
        preprocessor=lambda batch: batch,
        accelerator=DummyAccelerator(),
        max_batches=None,
    )

    assert metrics["loss"] == 2.0
    assert metrics["scalar_metric"] == 3.0
    assert metrics["vector_metric_0"] == 1.0
    assert metrics["vector_metric_1"] == 2.0
