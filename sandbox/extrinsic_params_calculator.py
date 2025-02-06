import numpy as np

# Function to compute the rotation matrix and translation vector
def compute_extrinsics(theta_deg, r):
    theta_rad = np.deg2rad(theta_deg)
    # R_1 = np.array([
    #     [np.cos(theta_rad), 0, -np.sin(theta_rad)],
    #     [0, 1, 0],
    #     [np.sin(theta_rad), 0, np.cos(theta_rad)]
    # ])
    # R_1 = np.array([ # constant wrt Z axis
    #     [np.cos(theta_rad), -np.sin(theta_rad), 0],
    #     [np.sin(theta_rad), np.cos(theta_rad), 0],
    #     [0, 0, 1]
        
    # ])
    # R_z = np.array([
    #     [0,1,0],
    #     [-1,0,0],
    #     [0,0,1]
    # ])
    # R_x = np.array([
    #     [1,0,0],
    #     [0,0,-1],
    #     [0,1,0]
    # ])
    # R_zx = np.dot(R_x,R_z)
    # R = np.dot(R_1,R_zx)


    # # t = np.array([
    # #     [r * np.cos(theta_rad)],
    # #     [0],
    # #     [r * np.sin(theta_rad)]
    # # ])
    # t = np.array([ # constant wrt Z axis
    #     [r * np.cos(theta_rad)],
    #     [r * np.sin(theta_rad)],
    #     [0]
    # ])

    R = np.array([
        [-np.sin(theta_rad), np.cos(theta_rad), 0],
        [0, 0, -1],
        [-np.cos(theta_rad), -np.sin(theta_rad), 0]
        ])
    t = np.array([r * np.cos(theta_rad), r * np.sin(theta_rad), 0])
    return R, t


# Function to combine rotation matrix and translation vector into a homogeneous extrinsic matrix
def extrinsic_to_homogeneous(R, t):
    homogeneous_matrix = np.eye(4)
    homogeneous_matrix[:3, :3] = R
    T = -np.dot(R, t)
    homogeneous_matrix[:3, 3] = T.flatten()
    return homogeneous_matrix

#covert world coordinate to camera coordinate
def world_to_camera_frame(P_batch, extrinsics):
    # For each point in the batch, apply the transformation
    transformed_batch = []
    for P in P_batch:
        transformed_points = []
        for extrinsic in extrinsics:
            # Convert P to homogeneous coordinates (x, y, z, 1)
            P_homogeneous = np.hstack([P, 1])
            # Apply transformation and extract the first 3 components
            transformed_points.append(np.dot(extrinsic, P_homogeneous)[:3])
        transformed_batch.append(transformed_points)
    
    return transformed_batch

#convert camera coordinate to world coordinate
def camera_to_world_frame(P, extrinsics):
    return [np.dot(np.linalg.inv(extrinsic), P) for extrinsic in extrinsics]
