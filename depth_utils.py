import numpy as np
import cv2
import matplotlib.pyplot as plt

def project_points_to_depth(pcd, intrinsic):
    # project open3d point cloud to numpy depth image
    points = np.asarray(pcd.points)
    camera_points = points @ intrinsic.intrinsic_matrix.T

    # normalize the camera points
    camera_points /= camera_points[:, 2:3]

    # find pixels within the image
    valid_mask = (camera_points[:, 0] >= 0) & (camera_points[:, 0] < intrinsic.width) & \
                 (camera_points[:, 1] >= 0) & (camera_points[:, 1] < intrinsic.height)
    depth = camera_points[:, 2] * valid_mask

    # reshape to 2D image
    depth_map = depth.reshape([intrinsic.height, intrinsic.width]).astype(np.float32)
    valid_mask = valid_mask.reshape([intrinsic.height, intrinsic.width]).astype(np.float32)

    return depth_map, valid_mask

def point_to_normal(pcd, height, width):
    points = np.asarray(pcd.points).reshape([height, width, 3])
    output = np.zeros([height, width, 3])
    dx = points[1:-1, 2:, ...] - points[1:-1, :-2, ...]
    dy = points[2:, 1:-1, ...] - points[:-2, 1:-1, ...]
    normal_map = np.cross(dx.reshape([-1, 3]), dy.reshape([-1, 3]))
    normal_map = normal_map / np.linalg.norm(normal_map, axis=1, keepdims=True)
    output[1:-1, 1:-1, ...] = normal_map.reshape([height-2, width-2, 3])
    return output

def compute_consistency(normal_map1, normal_map2):
    normal_consist = (normal_map1 * normal_map2).sum(axis=2)
    return normal_consist

def combine_depthmaps(depth1, depth2, mask, ths=0.5):

    ## DEBUG ##
    plt.figure()

    # initialize combined depth
    combined_depth = depth1.copy()
    # fill pixels with low consistency with depth2
    combined_depth[mask < ths] = depth2[mask < ths]
    plt.subplot(3, 1, 1)
    plt.imshow(combined_depth, cmap="gray")
    # apply bilateral filter to smooth the depth
    combined_depth = cv2.bilateralFilter(combined_depth, 5)
    plt.subplot(3, 1, 2)
    plt.imshow(combined_depth, cmap="gray")
    # remove impulse noise in depth1
    combined_depth = cv2.medianBlur(combined_depth, 5)
    plt.subplot(3, 1, 3)
    plt.imshow(combined_depth, cmap="gray")

    plt.show()
    ###############

    return combined_depth