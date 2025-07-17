# -*- coding: utf-8 -*-
"""
Created on Thu Jul 17 13:52:22 2025

@author: ChatGPT4o
"""

import numpy as np

def quaternion_rotate(point, axis, angle_rad):
    """
    Rotates a 3D point around a given axis using quaternion rotation.

    Args:
        point (np.ndarray): A 3D point as a numpy array [x, y, z].
        axis (np.ndarray): A unit vector representing the axis of rotation.
        angle_rad (float): The rotation angle in radians.

    Returns:
        np.ndarray: The rotated 3D point.
    """

    # Normalize the axis
    axis = axis / np.linalg.norm(axis)

    # Quaternion components
    w = np.cos(angle_rad / 2.0)
    x, y, z = axis * np.sin(angle_rad / 2.0)
    q = np.array([w, x, y, z])             # rotation quaternion
    q_conj = np.array([w, -x, -y, -z])     # conjugate quaternion

    # Represent the point as a quaternion (0 + xi + yj + zk)
    p = np.array([0, *point])

    # Perform the rotation: q * p * q_conj
    def quaternion_mult(a, b):
        w1, x1, y1, z1 = a
        w2, x2, y2, z2 = b
        return np.array([
            w1*w2 - x1*x2 - y1*y2 - z1*z2,
            w1*x2 + x1*w2 + y1*z2 - z1*y2,
            w1*y2 - x1*z2 + y1*w2 + z1*x2,
            w1*z2 + x1*y2 - y1*x2 + z1*w2
        ])

    rotated_p = quaternion_mult(quaternion_mult(q, p), q_conj)

    # Return only the vector part (x, y, z)
    return rotated_p[1:]


point = np.array([1.0, 0.0, 0.0])               # Point to rotate
axis = np.array([0.0, 0.0, 1.0])                # Rotate around Z-axis
angle_rad = np.pi / 2                           # 90 degrees

rotated_point = quaternion_rotate(point, axis, angle_rad)
print(rotated_point)  # Output should be approximately [0, 1, 0]
