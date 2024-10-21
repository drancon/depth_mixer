import os
import numpy as np
import cv2
import open3d as o3d
import argparse
import glob
import json
from tqdm import tqdm
import matplotlib.pyplot as plt

from depth_utils import project_points_to_depth, point_to_normal, compute_consistency, combine_depthmaps

def main(args):
    # get the list of depth files
    src_depth_files = sorted(glob.glob(os.path.join(args.src_dir, "*.npz")))
    ref_depth_files = sorted(glob.glob(os.path.join(args.ref_dir, "*.png")))

    assert len(ref_depth_files) == len(src_depth_files)

    # load the parameters
    with open(args.param_path, "r") as f:
        params = json.load(f)["camera"]

    fx = params["fx"]
    fy = params["fy"]
    cx = params["cx"]
    cy = params["cy"]
    scale = params["scale"]

    # initialize o3d intrinsic matrix
    intrinsic = o3d.camera.PinholeCameraIntrinsic(
        width=params["width"],
        height=params["height"],
        fx=fx,
        fy=fy,
        cx=cx,
        cy=cy
    )

    # create the output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # read depth files and get adjusted depth
    for ref_depth_file, src_depth_file in tqdm(zip(ref_depth_files, src_depth_files), total=len(ref_depth_files)):
        # read the depth files
        src_depth = np.load(src_depth_file)["depth"]
        ref_depth = cv2.imread(ref_depth_file, cv2.IMREAD_ANYDEPTH) / scale

        # convert to o3d images
        src_depth_o3d = o3d.geometry.Image(src_depth)
        ref_depth_o3d = o3d.geometry.Image(ref_depth)

        print(src_depth_o3d, ref_depth_o3d)

        # project the depth to 3D points
        src_points = o3d.geometry.PointCloud.create_from_depth_image(src_depth_o3d, intrinsic, project_valid_depth_only=True)
        ref_points = o3d.geometry.PointCloud.create_from_depth_image(ref_depth_o3d, intrinsic, project_valid_depth_only=True)

        # compute validity mask of ref depth
        valid_mask = (ref_depth.flatten() > args.min_depth) & (ref_depth.flatten() < args.max_depth)
        valid_indices = np.where(valid_mask)[0]

        # get the correspondences between the same indices
        corr = np.arange(src_points.shape[0]).reshape(-1, 1).repeat(2, axis=1)
        corr = corr[valid_indices]
        print(corr.shape, src_points.shape, ref_points.shape, valid_indices.shape)
        corr = o3d.utility.Vector2iVector(corr)

        # compute RANSAC GICP
        reg_p2p = o3d.pipelines.registration.registration_ransac_based_on_correspondence(
            src_points, ref_points, corr, 0.5,
            o3d.pipelines.registration.TransformationEstimationForGeneralizedICP(),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
            o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 0.9999))

        # get the transformation matrix
        trans_mat = reg_p2p.transformation

        # apply the transformation to the src points
        adjusted_points = src_points.transform(trans_mat)

        ## DEBUG ##
        # src points - red, ref points - blue, adjusted points - green
        src_viz = o3d.geometry.PointCloud()
        src_viz.points = o3d.utility.Vector3dVector(src_points.points)
        src_viz.paint_uniform_color([1, 0, 0])
        ref_viz = o3d.geometry.PointCloud()
        ref_viz.points = o3d.utility.Vector3dVector(ref_points.points)
        ref_viz.paint_uniform_color([0, 0, 1])
        adjusted_viz = o3d.geometry.PointCloud()
        adjusted_viz.points = o3d.utility.Vector3dVector(adjusted_points.points)
        adjusted_viz.paint_uniform_color([0, 1, 0])
        o3d.visualization.draw_geometries([src_viz, ref_viz, adjusted_viz],
                                          window_name="src_points - red, ref_points - blue, adjusted_points - green")
        ###############

        # project the points to 2D
        adjusted_depth, adj_mask = project_points_to_depth(adjusted_points, intrinsic)
        print(adj_mask.mean())

        # compute consistency between ref depth and adjusted depth by comparing normal maps
        ref_normal = point_to_normal(ref_points)
        adjusted_normal = point_to_normal(adjusted_points)
        consistency_map = compute_consistency(ref_normal, adjusted_normal)

        # merge masks
        merged_mask = valid_mask.astype(np.float32) * adj_mask.astype(np.float32) * consistency_map

        # combine adjusted depth and ref depth by filling holes and removing impulse noies in ref depth
        combined_depth = combine_depthmaps(ref_depth, adjusted_depth, merged_mask, args.ths)

        ## DEBUG ##
        plt.figure()
        plt.subplot(3, 1, 1)
        plt.imshow(ref_depth, cmap="gray")
        plt.subplot(3, 1, 2)
        plt.imshow(adjusted_depth, cmap="gray")
        plt.subplot(3, 1, 3)
        plt.imshow(combined_depth, cmap="gray")
        plt.show()
        ###############

        # save the combined depth
        output_path = os.path.join(args.output_dir, os.path.basename(ref_depth_file))
        cv2.imwrite(output_path, (combined_depth * scale).astype(np.uint16))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ref_dir", type=str, required=True)
    parser.add_argument("--src_dir", type=str, required=True)
    parser.add_argument("--param_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--max_depth", type=float, default=10.0)
    parser.add_argument("--min_depth", type=float, default=0.0)
    parser.add_argument("--ths", type=float, default=0.5)
    args = parser.parse_args()
    main(args)

