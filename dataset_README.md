# Dataset Notes: Autobrik/SO-ARM100-dump-pocket-cleaning2

Last update: 2026-02-15

This file centralizes dataset and training-context notes for paper updates.

## 1) Dataset identity and source

- Repo id: `Autobrik/SO-ARM100-dump-pocket-cleaning2`
- Local cache path:
  `~/.cache/huggingface/lerobot/Autobrik/SO-ARM100-dump-pocket-cleaning2`
- Metadata checked from:
  - `meta/info.json`
  - `meta/episodes/chunk-000/file-*.parquet`
  - `data/chunk-000/file-*.parquet`

## 2) Dataset size

- Total episodes: `162`
- Episode index range: `0..161`
- Total frames: `73944`
- FPS: `30`
- Split declared in metadata: `train: 0:162` (all episodes available)

## 3) Episode duration stats (episodes 0..161)

Duration computed as `timestamp_max - timestamp_min` per episode.

- Mean duration: `15.1815 s`
- Min duration: `10.4 s` (episode `151`)
- Max duration: `26.6 s` (episode `67`)
- Std duration: `2.6706 s`
- Median duration: `14.8167 s`
- IQR: `3.3667 s` (`Q1=13.2667 s`, `Q3=16.6333 s`)
- Total recorded duration: `2459.4 s` (`40.99 min`)

Frame stats per episode:

- Mean frames: `456.44`
- Min frames: `313` (episode `151`)
- Max frames: `799` (episode `67`)

## 4) Train/Val split used for benchmark runs

Split rule used by the training scripts:

- `TRAIN_RATIO=0.8`
- `SPLIT_SEED=2026`
- Episodes shuffled then split

Result for 162 episodes:

- Train episodes: `129`
- Val episodes: `33`

Train episodes:

```text
[1,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,21,23,24,27,29,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,54,55,56,58,59,60,62,63,65,66,67,68,69,70,71,72,74,75,76,77,78,79,82,83,84,85,86,88,90,92,93,94,95,97,98,99,100,102,103,104,105,106,108,109,110,111,113,114,116,117,118,119,120,121,122,123,124,126,127,129,130,132,133,134,135,136,137,138,139,141,143,144,145,147,148,149,151,152,154,155,156,157,159,160,161]
```

Val episodes:

```text
[0,2,19,20,22,25,26,28,30,53,57,61,64,73,80,81,87,89,91,96,101,107,112,115,125,128,131,140,142,146,150,153,158]
```

## 5) Evaluation note (test protocol)

There is no separate offline "test split" in this setup.

- During training: offline validation is done on the fixed `val` episodes above.
- Final benchmark/test: done with **real-world autonomous rollouts** on the robot.
- Recommendation for reproducibility: keep a fixed evaluation seed/order for rollouts
  (same run ordering across models and sessions).

## 6) ACT command used in benchmark run

Command provided for ACT run:

```bash
/venv/lerobot/bin/lerobot-train --dataset.repo_id=Autobrik/SO-ARM100-dump-pocket-cleaning2 --dataset.episodes=[1,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,21,23,24,27,29,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,54,55,56,58,59,60,62,63,65,66,67,68,69,70,71,72,74,75,76,77,78,79,82,83,84,85,86,88,90,92,93,94,95,97,98,99,100,102,103,104,105,106,108,109,110,111,113,114,116,117,118,119,120,121,122,123,124,126,127,129,130,132,133,134,135,136,137,138,139,141,143,144,145,147,148,149,151,152,154,155,156,157,159,160,161] --policy.type=act --policy.device=cuda --policy.push_to_hub=false --output_dir=outputs/train/benchmark/20260215_071412_benchmark_act --job_name=benchmark_act --seed=1000 --batch_size=16 --num_workers=4 --steps=14728 --eval_freq=-1 --log_freq=100 --save_freq=3682 --offline_val.enable=true --offline_val.episodes=[0,2,19,20,22,25,26,28,30,53,57,61,64,73,80,81,87,89,91,96,101,107,112,115,125,128,131,140,142,146,150,153,158] --offline_val.freq=1841 --offline_val.track_best_checkpoint=true --wandb.enable=true --wandb.project=dump-pocket-benchmark
```

## 7) Training context for 4-model contrast

Scripts used:

- `run_train_act.sh`
- `run_train_dp.sh`
- `run_train_pi05.sh`
- `run_train_smolvla.sh`

Common behavior across all scripts:

- Read `total_episodes` from metadata (not hardcoded).
- Build split from `TRAIN_RATIO` and `SPLIT_SEED`.
- Use `offline_val` with explicit val episodes.
- Compute steps from `TARGET_EPOCHS` unless overridden:
  - `OVERRIDE_STEPS` has priority.
  - `USE_REFERENCE_STEPS=true` uses model reference steps.
- Logging/checkpoint frequencies derived from `steps_per_epoch` unless manually set.

Model-specific defaults:

- ACT (`run_train_act.sh`)
  - `--policy.type=act`
  - `BATCH_SIZE=16`
  - `DROP_LAST_FRAMES=0`
  - `REFERENCE_STEPS=100000`
- Diffusion Policy (`run_train_dp.sh`)
  - `--policy.type=diffusion`
  - `BATCH_SIZE=32`
  - `DROP_LAST_FRAMES=7`
  - `REFERENCE_STEPS=200000`
- pi0.5 (`run_train_pi05.sh`)
  - `--policy.type=pi05`
  - `--policy.pretrained_path=lerobot/pi05_base`
  - `--policy.gradient_checkpointing=true`
  - `--policy.freeze_vision_encoder=true`
  - `--policy.train_expert_only=true`
  - `--policy.dtype=bfloat16`
  - `BATCH_SIZE=8`
  - `DROP_LAST_FRAMES=0`
  - `REFERENCE_STEPS=100000`
- SmolVLA (`run_train_smolvla.sh`)
  - `--policy.path=lerobot/smolvla_base`
  - `BATCH_SIZE=16`
  - `DROP_LAST_FRAMES=0`
  - `REFERENCE_STEPS=20000`

## 8) Paper update checklist

- Update dataset size from old value to:
  - `162 episodes`
- Update average episode duration to:
  - `15.18 s` (at 30 Hz, about `456` frames per episode)
- If reporting split counts with current setup:
  - `129 train / 33 val`
- Mention that final test is done with real robot rollouts (not a held-out offline test split).

## 9) Paper-ready additions (recommended)

### 9.1 Split balance (train vs val)

These numbers help show that the split is not duration-biased.

- Train (`129` episodes):
  - Total frames: `58888`
  - Total duration: `1958.633 s` (`32.644 min`)
  - Mean duration: `15.1832 s` (std `2.5960`)
  - Median duration: `14.8333 s`
  - IQR: `3.2667 s`
  - Min/Max duration: `10.4 s` / `26.6 s`
- Val (`33` episodes):
  - Total frames: `15056`
  - Total duration: `500.767 s` (`8.346 min`)
  - Mean duration: `15.1747 s` (std `2.9879`)
  - Median duration: `14.4000 s`
  - IQR: `4.2000 s`
  - Min/Max duration: `11.0667 s` / `21.4333 s`

### 9.2 Effective training budget per model (default scripts)

Assuming current defaults in `run_train_*.sh` and `TARGET_EPOCHS=4`.

- ACT:
  - `batch_size=16`, `drop_last_frames=0`
  - `train_samples=58888`
  - `steps_per_epoch=3681`
  - `train_steps=14724`
  - `save_freq=3681`, `val_freq=1841`
- Diffusion Policy:
  - `batch_size=32`, `drop_last_frames=7`
  - `train_samples=57985`
  - `steps_per_epoch=1813`
  - `train_steps=7252`
  - `save_freq=1813`, `val_freq=907`
- pi0.5:
  - `batch_size=8`, `drop_last_frames=0`
  - `train_samples=58888`
  - `steps_per_epoch=7361`
  - `train_steps=29444`
  - `save_freq=7361`, `val_freq=3681`
- SmolVLA:
  - `batch_size=16`, `drop_last_frames=0`
  - `train_samples=58888`
  - `steps_per_epoch=3681`
  - `train_steps=14724`
  - `save_freq=3681`, `val_freq=1841`

### 9.3 Dataset scope / limitations

- Single-task dataset: `total_tasks=1`
- Robot type: `so_follower`
- Modalities:
  - `observation.images.top` (RGB)
  - `observation.images.wrist` (RGB)
  - `observation.state` (proprioception)
- FPS: `30`
- Total recorded duration: `40.99 min`

### 9.4 Duration outliers

- Longest episode: `episode 67` with `26.6 s` (`799` frames)
- Shortest episode: `episode 151` with `10.4 s` (`313` frames)

### 9.5 Reproducibility statement (suggested)

- Fixed split generation: `TRAIN_RATIO=0.8`, `SPLIT_SEED=2026`
- Explicit train/val episode lists are archived in this file.
- Final testing is on real robot rollouts with fixed evaluation ordering.
- Keep command lines and run metadata (timestamp/output dir/git commit) per model.
