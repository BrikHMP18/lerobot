#!/usr/bin/env python

"""Piper robot kinematics using official AgileX DH parameters.

Provides forward and inverse kinematics for the 6-DOF Piper robot arm
using the official Denavit-Hartenberg parameters from AgileX SDK.
"""

import math

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

# Joint limits for Piper robot (radians)
PIPER_JOINT_LIMITS = [
    (-2.96706, 2.96706),  # joint0
    (-2.09440, 2.09440),  # joint1
    (-2.96706, 2.96706),  # joint2
    (-2.09440, 2.09440),  # joint3
    (-2.96706, 2.96706),  # joint4
    (-6.28319, 6.28319),  # joint5
]

# IK convergence threshold (mm equivalent error)
IK_ERROR_THRESHOLD = 10.0


class PiperForwardKinematics:
    """Forward Kinematics using official AgileX Piper DH parameters.

    Implements the Denavit-Hartenberg convention for 6-DOF Piper robot arm.
    """

    def __init__(self, dh_is_offset: int = 0x01):
        """Initialize FK with official DH parameters.

        Args:
            dh_is_offset: DH version (0x00=old firmware <S-V1.6-3, 0x01=new >=S-V1.6-3)
        """
        self.RADIAN = 180 / math.pi
        self.PI = math.pi

        # Denavit-Hartenberg parameters for each link
        # _a: link lengths (mm)
        # _alpha: link twists (rad)
        # _theta: joint offsets (rad)
        # _d: link offsets (mm)
        if dh_is_offset == 0x01:
            # New DH parameters (firmware >= S-V1.6-3)
            self._a = [0, 0, 285.03, -21.98, 0, 0]
            self._alpha = [0, -self.PI / 2, 0, self.PI / 2, -self.PI / 2, self.PI / 2]
            self._theta = [0, -self.PI * 172.22 / 180, -102.78 / 180 * self.PI, 0, 0, 0]
            self._d = [123, 0, 0, 250.75, 0, 91]
            self.init_pos = [56.128, 0.0, 213.266, 0.0, 85.0, 0.0]  # xyz-mm, rpy-degree
        else:
            # Old DH parameters
            self._a = [0, 0, 285.03, -21.98, 0, 0]
            self._alpha = [0, -self.PI / 2, 0, self.PI / 2, -self.PI / 2, self.PI / 2]
            self._theta = [0, -self.PI * 174.22 / 180, -100.78 / 180 * self.PI, 0, 0, 0]
            self._d = [123, 0, 0, 250.75, 0, 91]
            self.init_pos = [55.0, 0.0, 205.0, 0.0, 85.0, 0.0]  # xyz-mm, rpy-degree

    def _matrix_to_euler(self, transform_matrix):
        """Convert transformation matrix to Euler angles.

        Args:
            transform_matrix: Flattened 4x4 transformation matrix (16 elements)

        Returns:
            [x, y, z, roll, pitch, yaw] (mm, degrees)
        """
        pos = [0.0] * 6
        # Extract position (x, y, z)
        pos[0] = transform_matrix[3]  # x position
        pos[1] = transform_matrix[7]  # y position
        pos[2] = transform_matrix[11]  # z position

        # Calculate Euler angles (roll, pitch, yaw) based on rotation matrix
        if transform_matrix[8] < -1 + 0.0001:
            pos[4] = self.PI / 2 * self.RADIAN  # pitch (beta)
            pos[5] = 0
            pos[3] = math.atan2(transform_matrix[1], transform_matrix[5]) * self.RADIAN  # roll (alpha)
        elif transform_matrix[8] > 1 - 0.0001:
            pos[4] = -self.PI / 2 * self.RADIAN  # pitch (beta)
            pos[5] = 0
            pos[3] = -math.atan2(transform_matrix[1], transform_matrix[5]) * self.RADIAN  # roll (alpha)
        else:
            # General case for Euler angles computation
            beta = math.atan2(
                -transform_matrix[8], math.sqrt(transform_matrix[0] ** 2 + transform_matrix[4] ** 2)
            )
            pos[4] = beta * self.RADIAN
            pos[5] = (
                math.atan2(transform_matrix[4] / math.cos(beta), transform_matrix[0] / math.cos(beta))
                * self.RADIAN
            )
            pos[3] = (
                math.atan2(transform_matrix[9] / math.cos(beta), transform_matrix[10] / math.cos(beta))
                * self.RADIAN
            )

        return pos

    def _matrix_multiply(self, matrix1, matrix2, rows, cols_rows, cols):
        """Multiply two matrices (flattened representation).

        Args:
            matrix1: First matrix (rows x cols_rows)
            matrix2: Second matrix (cols_rows x cols)
            rows, cols_rows, cols: Matrix dimensions

        Returns:
            Result matrix (rows x cols, flattened)
        """
        matrix_out = [0.0] * (rows * cols)
        for i in range(rows):
            for j in range(cols):
                tmp = 0.0
                for k in range(cols_rows):
                    tmp += matrix1[cols_rows * i + k] * matrix2[cols * k + j]
                matrix_out[cols * i + j] = tmp
        return matrix_out

    def _link_transformation(self, alpha, a, theta, d):
        """Compute DH transformation matrix for a single link.

        Args:
            alpha: Link twist (rad)
            a: Link length (mm)
            theta: Joint angle (rad)
            d: Link offset (mm)

        Returns:
            Flattened 4x4 transformation matrix
        """
        # Precompute trigonometric functions for efficiency
        calpha = math.cos(alpha)
        salpha = math.sin(alpha)
        ctheta = math.cos(theta)
        stheta = math.sin(theta)

        transform = [0.0] * 16  # 4x4 transformation matrix
        transform[0] = ctheta
        transform[1] = -stheta
        transform[2] = 0
        transform[3] = a

        transform[4] = stheta * calpha
        transform[5] = ctheta * calpha
        transform[6] = -salpha
        transform[7] = -salpha * d

        transform[8] = stheta * salpha
        transform[9] = ctheta * salpha
        transform[10] = calpha
        transform[11] = calpha * d

        transform[12] = 0
        transform[13] = 0
        transform[14] = 0
        transform[15] = 1

        return transform

    def calc_fk(self, cur_j):
        """Calculate forward kinematics for given joint configuration.

        Args:
            cur_j: Joint angles (rad) [joint0...joint5]

        Returns:
            TCP pose [x, y, z, roll, pitch, yaw] (mm, degrees)
        """
        # Initialize transformation matrices
        link_transforms = [[0.0] * 16 for _ in range(6)]

        # Compute the individual transformation matrices
        for i in range(6):
            c_theta = cur_j[i] + self._theta[i]
            link_transforms[i] = self._link_transformation(self._alpha[i], self._a[i], c_theta, self._d[i])

        # Multiply transformation matrices
        r02 = self._matrix_multiply(link_transforms[0], link_transforms[1], 4, 4, 4)
        r03 = self._matrix_multiply(r02, link_transforms[2], 4, 4, 4)
        r04 = self._matrix_multiply(r03, link_transforms[3], 4, 4, 4)
        r05 = self._matrix_multiply(r04, link_transforms[4], 4, 4, 4)
        r06 = self._matrix_multiply(r05, link_transforms[5], 4, 4, 4)

        # Extract final TCP pose
        tcp_pose = self._matrix_to_euler(r06)
        return tcp_pose


class PiperInverseKinematics:
    """Numerical IK solver for Piper robot using scipy optimization.

    Uses least-squares optimization to solve IK, respecting joint limits.
    Convergence is typically achieved in 20-50ms per pose.
    """

    def __init__(self, dh_is_offset: int = 0x01):
        """Initialize IK solver.

        Args:
            dh_is_offset: DH version (0x00=old, 0x01=new firmware)
        """
        self.fk = PiperForwardKinematics(dh_is_offset)
        self.joint_limits = PIPER_JOINT_LIMITS

    def _tcp_to_fk_format(self, tcp_pose):
        """Convert LeRobot TCP format to FK format.

        Args:
            tcp_pose: [x, y, z, rx, ry, rz] (meters, axis-angle radians)

        Returns:
            [x, y, z, roll, pitch, yaw] (mm, Euler degrees)
        """
        # Convert position from meters to mm
        x_mm = tcp_pose[0] * 1000
        y_mm = tcp_pose[1] * 1000
        z_mm = tcp_pose[2] * 1000

        # Convert axis-angle to Euler angles
        axis_angle = tcp_pose[3:6]
        if np.linalg.norm(axis_angle) < 1e-6:
            roll, pitch, yaw = 0.0, 0.0, 0.0
        else:
            euler_rad = Rotation.from_rotvec(axis_angle).as_euler("xyz", degrees=False)
            roll, pitch, yaw = euler_rad * 180 / math.pi

        return [x_mm, y_mm, z_mm, roll, pitch, yaw]

    def solve_ik(self, target_tcp, initial_joints=None, max_iterations=100):
        """Solve IK using least-squares numerical optimization.

        Args:
            target_tcp: Target TCP pose [x, y, z, rx, ry, rz] (meters, axis-angle rad)
            initial_joints: Initial joint guess (rad). Defaults to zeros.
            max_iterations: Max optimization iterations

        Returns:
            (joints, success, error):
                - joints: Solution [joint0...joint5] (rad)
                - success: True if converged
                - error: Final error magnitude (mm equivalent)
        """
        if initial_joints is None:
            initial_joints = np.zeros(6)

        # Convert target to FK format
        target_fk = self._tcp_to_fk_format(target_tcp)

        def error_function(joints):
            """Compute weighted error between current FK and target."""
            joints_clipped = np.clip(
                joints, [lim[0] for lim in self.joint_limits], [lim[1] for lim in self.joint_limits]
            )
            current_fk = self.fk.calc_fk(joints_clipped.tolist())

            # Position error (mm)
            pos_error = np.array(current_fk[:3]) - np.array(target_fk[:3])

            # Orientation error (degrees), normalized to [-180, 180]
            ori_error = np.array(current_fk[3:6]) - np.array(target_fk[3:6])
            ori_error = (
                np.arctan2(np.sin(ori_error * math.pi / 180), np.cos(ori_error * math.pi / 180))
                * 180
                / math.pi
            )

            # Weight position 10x more than orientation
            return np.concatenate([pos_error, ori_error * 0.1])

        # Solve using least squares optimization
        result = least_squares(
            error_function,
            initial_joints,
            bounds=([lim[0] for lim in self.joint_limits], [lim[1] for lim in self.joint_limits]),
            max_nfev=max_iterations,
            ftol=1e-6,
            xtol=1e-6,
        )

        final_error = np.linalg.norm(result.fun)
        success = result.success and final_error < IK_ERROR_THRESHOLD
        return result.x, success, final_error
