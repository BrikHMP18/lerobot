# UMI Zarr → LeRobot Conversion

## Overview

Scripts to convert UMI `.zarr` datasets to LeRobot format for HuggingFace Hub upload.

**Two conversion modes:**

- **TCP Pose** (cartesian): ✅ Working - `[x, y, z, rx, ry, rz, gripper]`
- **Joint Angles** (articular): ⚠️ Blocked - requires valid URDF

---

## Quick Start (TCP Pose - Working)

```bash
conda activate lerobot
cd ~/NONHUMAN/lerobot
./examples/port_datasets_umi/convert_umi_tcp.sh
```

**Output**: Dataset uploaded to `NONHUMAN-RESEARCH/pick_the_cup_demo_dataset`

---

## Joint Angles Conversion (Blocked)

### Current Status

❌ **IK solver fails** with available Piper URDFs:

- Tested: `piper_description.urdf` (new) and `piper_description_old.urdf`
- Problem: Neither URDF produces valid IK solutions for simple poses
- Root cause: Incorrect joint transformations (RPY), offsets, or limits in URDF files

### Required Next Steps

**PRIMARY TASK: Obtain validated URDF**

1. **Contact AgileX Robotics**
   - Request calibrated URDF for Piper robot
   - Verify it matches your firmware version

2. **Alternative: Generate URDF from real robot**
   - Use MoveIt Setup Assistant
   - Validate FK/IK with known poses from dataset

3. **Test validation command:**

```bash
conda create -n ik_test python=3.10 -y
conda activate ik_test
pip install "numpy<2" roboticstoolbox-python spatialmath-python scipy zarr

python -c "
import roboticstoolbox as rtb
from spatialmath import SE3
import numpy as np

robot = rtb.Robot.URDF('/path/to/piper.urdf')
ee = [i for i, l in enumerate(robot.links) if l.name == 'gripper_base'][0]
pose = SE3.Rt(np.eye(3), np.array([0.3, 0.0, 0.2]))
sol = robot.ikine_LM(pose, end=robot.links[ee])
print(f'IK: {\"SUCCESS\" if sol.success else \"FAIL\"}')
"
```

✅ **Valid URDF = IK returns SUCCESS**

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
    ├── port_umi_zarr_tcp.py      # ✅ Working
    ├── port_umi_zarr_joints.py   # ⚠️ Blocked (needs URDF)
    ├── convert_umi_tcp.sh         # ✅ Working
    ├── convert_umi_joints.sh      # ⚠️ Blocked (needs URDF)
    └── README_ZARR2LEROBOT.md
```

---

## Technical Notes

### Why TCP Pose Works

- UMI datasets store TCP pose natively (no conversion needed)
- Robot SDK handles IK during deployment
- Compatible with any robot (robot-agnostic)

### Why Joint Angles Is Blocked

- Requires accurate URDF for inverse kinematics
- Current URDFs fail IK solver validation
- `roboticstoolbox-python` also conflicts with LeRobot's NumPy 2.x
- **Solution**: Get validated URDF from manufacturer or calibrate from real robot

### Data Normalization

- Dataset stores raw values (meters, radians)
- LeRobot applies normalization during training via `NormalizerProcessorStep`
- Statistics auto-calculated by `dataset.finalize()`

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

- ✅ **TCP conversion**: Fully working, use `./examples/port_datasets_umi/convert_umi_tcp.sh`
- ⚠️ **Joint conversion**: Blocked until valid Piper URDF obtained
- 🎯 **Next action**: Contact AgileX Robotics or generate URDF from real robot

---

## Contact

For URDF validation or joint conversion issues, prioritize obtaining manufacturer-validated robot model.
