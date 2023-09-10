import torch
from geometry_msgs.msg import Pose
from scipy.spatial.transform import Rotation

import pdb

"""
Utilities for converting ROS2 pose messages to torch tensors, and for converting
poses expressed in other coordinate systems to the Nerfstudio coordinate sytem.

Useful documentation:
    1. https://docs.nerf.studio/en/latest/quickstart/data_conventions.html
    2. https://github.com/UZ-SLAMLab/ORB_SLAM3/blob/master/Calibration_Tutorial.pdf
"""


def ros_pose_to_homogenous(pose_message: Pose):
    """
    Converts a ROS2 Pose message to a 4x4 homogenous transformation matrix
    as a torch tensor (half precision).
    """
    quat = pose_message.pose.orientation
    pose = pose_message.pose.position

    R = Rotation.from_quat([quat.x, quat.y, quat.z, quat.w]).as_matrix()
    t = torch.Tensor([pose.x, pose.y, pose.z])

    T = torch.eye(4)
    T[:3, :3] = torch.from_numpy(R)
    T[:3, 3] = t
    return T.to(dtype=torch.float32)


def cuvslam_to_nerfstudio(T_cuvslam: torch.Tensor):
    """
    Converts a homogenous matrix 4x4 from the coordinate system used by the odometry topic
    in ISAAC ROS VSLAM to the Nerfstudio camera coordinate system 3x4 matrix.

    Equivalent operation to:
        T_out = T_in @ (Rotate Z by 180) @ (Rotate Y by 90) @ (Rotate Z by 90)
    """
    T_ns = T_cuvslam[:, [1, 2, 0, 3]]
    T_ns[:, [0, 2]] *= -1
    return T_ns[:3, :]


def orbslam3_to_nerfstudio(T_orbslam3: torch.Tensor):
    """
    Converts a homogenous matrix 4x4 from the coordinate system used in orbslam3
    to the Nerfstudio camera coordinate system 3x4 matrix.

    Equivalent operation to:
        T_out = (Y by 180) @ (Z by 180) @ (X by 90) @ T_in @ (X @ 180)
    """
    T_ns = T_orbslam3[[0, 2, 1, 3], :]
    T_ns[:, [1, 2]] *= -1
    T_ns[2, :] *= -1
    return T_ns[:3, :]
