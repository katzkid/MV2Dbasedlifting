import numpy as np

# Function to compute the rotation matrix and translation vector
def compute_extrinsics(theta_deg, r):
    theta_rad = np.deg2rad(theta_deg)
    R = np.array([
        [np.cos(theta_rad), 0, np.sin(theta_rad)],
        [0, 1, 0],
        [-np.sin(theta_rad), 0, np.cos(theta_rad)]
    ])
    t = np.array([
        [r * np.cos(theta_rad)],
        [0],
        [r * np.sin(theta_rad)]
    ])
    return R, t


# Function to combine rotation matrix and translation vector into a homogeneous extrinsic matrix
def extrinsic_to_homogeneous(R, t):
    homogeneous_matrix = np.eye(4)
    homogeneous_matrix[:3, :3] = R
    homogeneous_matrix[:3, 3] = t.flatten()
    return homogeneous_matrix
