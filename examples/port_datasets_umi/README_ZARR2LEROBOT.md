# UMI Zarr → LeRobot Conversion

## Overview

Scripts to convert UMI `.zarr` datasets to LeRobot format for HuggingFace Hub upload.

**Three conversion modes:**

- **TCP Pose** (cartesian): ✅ Working - `[x, y, z, rx, ry, rz, gripper]`
- **TCP → Joint Angles** (conversion): ✅ Working - Convert existing TCP dataset to joints
- **Direct Joint Angles** (from .zarr): ⚠️ Blocked - requires valid URDF

---

## Quick Start (TCP Pose - Working)

```bash
conda activate lerobot
cd ~/NONHUMAN/lerobot
./examples/port_datasets_umi/convert_umi_tcp.sh
```

**Output**: Dataset uploaded to `NONHUMAN-RESEARCH/pick_the_cup_demo_dataset`

---

## TCP → Joint Angles Conversion (Working)

### Quick Start

Convert an existing TCP dataset to joint angles:

```bash
conda activate lerobot
cd ~/NONHUMAN/lerobot
./examples/port_datasets_umi/convert_tcp_to_joints.sh
```

**Input**: TCP dataset at `examples/port_datasets_umi/pick_the_cup_demo_dataset`
**Output**: Joint angles dataset uploaded to `NONHUMAN-RESEARCH/pick_the_cup_demo_dataset_joints`

- Uses numerical IK solver with official Piper DH parameters
- No URDF required ✅

### Configuration

Edit `convert_tcp_to_joints.sh`:

```bash
INPUT_DATASET="path/to/tcp_dataset"
OUTPUT_REPO_ID="your-org/dataset_joints"
DH_IS_OFFSET=1  # 0=old DH, 1=new DH (firmware >= S-V1.6-3)
```

---

## Configuration

### TCP Conversion (convert_umi_tcp.sh)

```bash
ZARR_PATH="$HOME/NONHUMAN/universal_manipulation_interface/example_demo_session/dataset.zarr"
REPO_ID="NONHUMAN-RESEARCH/pick_the_cup_demo_dataset"
OUTPUT_DIR="$HOME/NONHUMAN/umi_lerobot_datasets/pick_the_cup_demo_dataset"
FPS="30"
TASK_DESC="pick up the cup and put it in the plate"
```

### Joint Conversion (convert_umi_joints.sh - blocked)

```bash
# Same as above, plus:
ROBOT_TYPE="piper"
URDF_PATH="$HOME/NONHUMAN/piper_urdf/piper_description.urdf"  # Needs valid URDF
```

---

## Files

```
lerobot/
└── examples/port_datasets_umi/
    ├── port_umi_zarr_tcp.py        # ✅ .zarr → TCP pose
    ├── convert_tcp_to_joints.py    # ✅ TCP → Joint angles (NEW)
    ├── port_umi_zarr_joints.py     # ⚠️ .zarr → Joints (blocked, needs URDF)
    ├── convert_umi_tcp.sh           # ✅ Run TCP conversion
    ├── convert_tcp_to_joints.sh     # ✅ Run TCP→Joints conversion (NEW)
    ├── convert_umi_joints.sh        # ⚠️ Run direct joints (blocked)
    └── README_ZARR2LEROBOT.md
```

---

## Technical Notes

### Notes

- TCP pose: UMI stores natively, robot-agnostic
- Direct joint conversion: Blocked (requires valid URDF)
- Data normalization: Applied automatically by LeRobot during training

---

## Troubleshooting

| Issue                                       | Solution                                      |
| ------------------------------------------- | --------------------------------------------- |
| `codec not available: 'imagecodecs_jpegxl'` | `pip install imagecodecs-numcodecs`           |
| `FileExistsError` on output dir             | `rm -rf $OUTPUT_DIR`                          |
| `403 Forbidden` HuggingFace                 | Check `huggingface-cli whoami` for org access |
| IK fails with URDF                          | **Contact AgileX for validated URDF**         |

---

## Summary

- ✅ **TCP conversion** (.zarr → TCP): Fully working, use `./examples/port_datasets_umi/convert_umi_tcp.sh`
- ✅ **TCP → Joints conversion**: Fully working, use `./examples/port_datasets_umi/convert_tcp_to_joints.sh`
- ⚠️ **Direct joint conversion** (.zarr → Joints): Blocked until valid Piper URDF obtained
- 🎯 **Recommended workflow**: .zarr → TCP → Joints (two-step conversion)
