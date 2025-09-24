import csv
import glob
import json
import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Tuple

import cv2
import loguru
import numpy as np
import open3d as o3d
import tqdm
from scipy.spatial.transform import Rotation
from scipy.spatial.transform import Rotation as R


class Arti4DDataset:
    """Arti4DDataset class.
    This class implements a dataset class for Azure Kinect RGB-D data.
    Returns a dictionary containing the keys "rgb", "depth", "pose", "pcd", "mask", and "idx".

    Args:
        RGBDDataset (dict): Configuration dictionary.
    """

    def __init__(self, cfg):
        self.calc_pcd = cfg.get("calc_pcd", False)
        self.depth_min = cfg.get("depth_min", 0.0)
        self.depth_max = cfg.get("depth_max", 10.0)
        self.root_path = cfg["root_path"]
        self.rgb_path = cfg["root_path"] / "rgb"
        self.depth_path = cfg["root_path"] / "depth"
        self.camera_poses_path = cfg["root_path"] / "odom"
        self.tf_file_path = cfg.get("tf_file_path", None)
        self.flipped = cfg.get("flipped", None)
        self.gt_poses = cfg.get("gt_poses", None)
        self.droid_slam = cfg.get("droid_slam", False)
        self.data_list = self._get_data_list()
        self._load_parameters()

    def _load_parameters(self):
        self.rgb_intrinsics = np.array(
            self._read_camera_params(self.rgb_path / "camera_info.txt")["K"]
        ).reshape(3, 3)
        self.depth_intrinsics = np.array(
            self._read_camera_params(self.depth_path / "camera_info.txt")["K"]
        ).reshape(3, 3)
        if self.camera_poses_path and self.gt_poses:
            self.camera_poses = self._read_camera_poses(
                self.camera_poses_path / Path(self.root_path.stem + ".csv")
            )
        elif self.droid_slam:
            self.camera_poses = self._read_camera_poses(
                self.root_path / "cam_trajectory.csv"
            )
            loguru.logger.info(
                f"Loaded DroidSLAM camera poses from: {self.root_path / 'cam_trajectory.csv'}"
            )
            with open(self.root_path / "registration_matrix.json", "r") as f:
                reg_matrix = json.load(f)
                self.reg_matrix = np.array(reg_matrix["transformation_matrix"]).reshape(
                    4, 4
                )
        if self.tf_file_path is not None and os.path.exists(self.tf_file_path):
            self._read_tf()
        else:
            loguru.logger.error(
                "Failed to load TF calibration file. Please check if the file exists and the path is correct: %s",
                self.tf_file_path,
            )
            raise Exception("TF calibration file is missing or incorrect.")
        if Path(
            os.path.join(self.root_path, Path(self.root_path).stem + ".json")
        ).exists():
            self._read_scene_articulation(
                os.path.join(self.root_path, Path(self.root_path).stem + ".json")
            )
            loguru.logger.info(
                f"Loaded scene articulation file:{os.path.join(self.root_path, Path(self.root_path).stem + '.json')}"
            )
        else:
            loguru.logger.warning(
                "Failed to load scene articulation file. Please check if the file exists and the path is correct: %s",
                os.path.join(self.root_path, Path(self.root_path).stem + ".json"),
            )

    def _read_camera_params(self, file_path: Path) -> Dict[str, Any]:
        """
        Read camera parameters from a text file.

        Args:
            file_path (Path): Path to the text file.

        Returns:
            Dict[str, Any]: Dictionary containing the camera parameters.
        """
        self.camera_params = {}
        with open(file_path, "r") as file:
            for line in file:
                if ":" in line:
                    key, value = map(str.strip, line.split(":", 1))
                    if key in ["D", "K", "R", "P"]:
                        self.camera_params[key] = tuple(
                            map(float, value.strip("()").split(","))
                        )
                    elif key in [
                        "binning_x",
                        "binning_y",
                        "width",
                        "height",
                        "x_offset",
                        "y_offset",
                        "seq",
                        "secs",
                        "nsecs",
                    ]:
                        # ignore if key already exists
                        if key in self.camera_params:
                            continue
                        self.camera_params[key] = int(value)
                    elif key == "do_rectify":
                        self.camera_params[key] = value.lower() == "true"
                    else:
                        self.camera_params[key] = value
        return self.camera_params

    def _read_scene_articulation(self, path):
        """
        Read articulated objects information from a JSON file.

        Args:
            path (str): Path to the JSON file containing articulation data.

        The JSON file should contain a dictionary where keys are object names and values
        are dictionaries with 'position' and 'axis' keys, each containing a list of 3 coordinates.
        """
        try:
            with open(path, "r") as f:
                self.articulated_objects = json.load(f)

            # Basic structure validation
            if not isinstance(self.articulated_objects, dict):
                loguru.logger.warning(
                    f"Articulation data is not in the expected dictionary format"
                )
                self.articulated_objects = {}
            else:
                loguru.logger.info(
                    f"Successfully loaded {len(self.articulated_objects)} articulated objects from {path}"
                )
        except Exception as e:
            loguru.logger.error(f"Failed to load articulated objects from {path}: {e}")
            self.articulated_objects = {}

    def _read_camera_poses(self, path):
        if not os.path.exists(path):
            loguru.logger.warning(f"Camera poses file not found at {path}")
            return None
        with open(path, "r") as file:
            reader = csv.DictReader(file)
            camera_poses = []
            for row in reader:
                # Strip whitespace from keys
                row = {k.strip(): v for k, v in row.items()}
                camera_pose = {
                    "timestamp": float(row["timestamp"]),
                    "x": float(row["x"]),
                    "y": float(row["y"]),
                    "z": float(row["z"]),
                    "qx": float(row["qx"]),
                    "qy": float(row["qy"]),
                    "qz": float(row["qz"]),
                    "qw": float(row["qw"]),
                }
                if self.droid_slam:
                    camera_pose["timestamp"] = float(
                        str(int(camera_pose["timestamp"]))[:10]
                        + "."
                        + str(int(camera_pose["timestamp"]))[10:]
                    )
                    # as it saved in the wrong order
                    (
                        camera_pose["qy"],
                        camera_pose["qz"],
                        camera_pose["qw"],
                        camera_pose["qx"],
                    ) = (
                        camera_pose["qx"],
                        camera_pose["qy"],
                        camera_pose["qz"],
                        camera_pose["qw"],
                    )
                camera_poses.append(camera_pose)
        return camera_poses

    def _find_closest_pose(self, data, target_timestamp):
        return min(data, key=lambda x: abs(x["timestamp"] - target_timestamp))

    def _pose_to_se3(self, pose):
        x, y, z = pose["x"], pose["y"], pose["z"]
        qx, qy, qz, qw = pose["qx"], pose["qy"], pose["qz"], pose["qw"]
        rotation = R.from_quat([qx, qy, qz, qw])
        R_matrix = rotation.as_matrix()
        # Construct the SE3 matrix
        SE3 = np.eye(4)
        SE3[:3, :3] = R_matrix
        SE3[:3, 3] = [x, y, z]
        return SE3

    def _get_closest_pose_in_se3(self, target_timestamp):
        target_timestamp = float(
            str(int(target_timestamp))[:10] + "." + str(int(target_timestamp))[10:]
        )
        closest_pose = self._find_closest_pose(self.camera_poses, target_timestamp)
        se3_matrix = self._pose_to_se3(closest_pose)
        return se3_matrix

    def _get_data_list(self):
        rgb_files = glob.glob(str(Path(self.root_path) / "rgb" / "*.jpg"))
        depth_files = glob.glob(str(Path(self.root_path) / "depth" / "*.png"))
        rgb_files.sort()
        depth_files.sort()
        return list(zip(rgb_files, depth_files))

    def __getitem__(self, idx):
        rgb_path, depth_path = self.data_list[idx]
        rgb = self._load_image(rgb_path)
        depth = self._load_depth(depth_path)
        timestamp = float(Path(rgb_path).stem.split("_")[-1])
        pose = (
            self._get_closest_pose_in_se3(timestamp) if self.camera_poses else np.eye(4)
        )
        if self.droid_slam:
            pose = self.reg_matrix @ pose  # apply the registration matrix
        if self.gt_poses:  # poses generated from the HTC tracking system
            pose = pose @ self.base_T_depth @ self.depth_T_rgb
        if self.flipped:  # input is flipped
            T_flip = np.eye(4)
            T_flip[:3, :3] = Rotation.from_euler("z", 180, degrees=True).as_matrix()
            pose = pose @ T_flip

        sample = {
            "rgb": rgb,
            "depth": depth.astype(np.float32),
            "pose": pose,
            "idx": idx,
        }
        return sample

    def __len__(self):
        return len(self.data_list)

    def _load_image(self, path):
        rgb = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
        return rgb

    def _load_depth(self, path):
        depth = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        return depth

    def get_scene_articulated_objects(self):
        """
        Returns the articulated objects in the scene.

        Returns:
            dict: A dictionary containing the articulated objects.
        """
        return self.articulated_objects

    def create_pcd(self, rgb_image, depth_image, camera_pose=None, maskout=None):
        """
        Creates a point cloud from RGB and depth images.

        This function generates a point cloud using the provided RGB and depth images.
        Optionally, a camera pose can be applied to transform the point cloud, and a mask
        can be used to filter out specific regions.

        Args:
            rgb_image (np.ndarray): The RGB image.
            depth_image (np.ndarray): The depth image.
            camera_pose (np.ndarray, optional): The camera pose. Defaults to None.
            maskout (np.ndarray, optional): The mask to filter out regions. Defaults to None.

        Returns:
            tuple: A tuple containing the point cloud and the valid mask.
        """
        # assert depth_image.shape == rgb_image.shape[:2], "Depth and RGB image dimensions do not match"

        # Convert depth to meters.
        z = depth_image / 1000.0

        # Create mask for valid depth values within the specified range.
        valid_mask = (z > self.depth_min) & (z < self.depth_max)
        if maskout is not None:
            valid_mask &= ~maskout

        # Get valid pixel indices (row, col).
        valid_indices = np.nonzero(
            valid_mask
        )  # valid_indices[0] = y, valid_indices[1] = x

        # Compute valid depth values.
        z_valid = z[valid_indices]
        x_valid = (
            (valid_indices[1] - self.depth_intrinsics[0, 2])
            * z_valid
            / self.depth_intrinsics[0, 0]
        )
        y_valid = (
            (valid_indices[0] - self.depth_intrinsics[1, 2])
            * z_valid
            / self.depth_intrinsics[1, 1]
        )

        # Stack to form (N, 3) point coordinates.
        points = np.column_stack((x_valid, y_valid, z_valid))

        # Apply camera pose transformation if provided.
        if camera_pose is not None:
            # print("Camera pose:", camera_pose)
            points = (camera_pose[:3, :3] @ points.T).T + camera_pose[:3, 3]

        # Extract color information.
        if rgb_image.ndim == 3:
            colors = rgb_image[valid_indices] / 255.0
        else:
            colors = np.zeros((points.shape[0], 3))

        # Create Open3D point cloud.
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(colors)
        return pcd, valid_mask

    def _read_tf(self):
        with open(self.tf_file_path, "r") as f:
            try:
                tf = json.load(f)
                if (
                    not all(key in tf for key in ["depth_T_rgb", "base_T_depth"])
                    or not all(
                        key in tf["depth_T_rgb"] for key in ["rotation", "translation"]
                    )
                    or not all(
                        key in tf["base_T_depth"] for key in ["rotation", "translation"]
                    )
                ):
                    raise ValueError("Invalid TF file structure.")

                quat = np.array(list(tf["depth_T_rgb"]["rotation"].values()))
                trans = np.array(list(tf["depth_T_rgb"]["translation"].values()))
                self.depth_T_rgb = np.eye(4)
                self.depth_T_rgb[:3, :3] = Rotation.from_quat(quat).as_matrix()
                self.depth_T_rgb[:3, 3] = trans

                quat = np.array(list(tf["base_T_depth"]["rotation"].values()))
                trans = np.array(list(tf["base_T_depth"]["translation"].values()))
                self.base_T_depth = np.eye(4)
                self.base_T_depth[:3, :3] = Rotation.from_quat(quat).as_matrix()
                self.base_T_depth[:3, 3] = trans
            except (KeyError, ValueError) as e:
                loguru.logger.error(f"Error reading TF file: {e}")
                raise Exception("TF file is malformed or missing required keys.")

    def get_scene_tsdf(
        self,
        voxel_size: float = 0.02,
        truncation_distance: float = 0.03,
    ) -> o3d.pipelines.integration.TSDFVolume:
        """
        Aggregates the RGB-D images in the dataset into a TSDF volume.

        Args:
            voxel_size (float): The size of each voxel in meters.
            truncation_distance (float): The distance (in meters) within which
                depth values are considered for surface reconstruction.
            volume_bound_min (Tuple[float, float, float]): The minimum bounds of the
                TSDF volume in world coordinates.
            volume_bound_max (Tuple[float, float, float]): The maximum bounds of the
                TSDF volume in world coordinates.

        Returns:
            o3d.pipelines.integration.TSDFVolume: The aggregated TSDF volume.
        """
        volume = o3d.pipelines.integration.ScalableTSDFVolume(
            voxel_length=voxel_size,
            sdf_trunc=truncation_distance,
            color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8,
        )

        for i in tqdm.tqdm(range(len(self)), desc="Integrating frames into TSDF"):
            sample = self[i]
            rgb_array = sample["rgb"].astype(np.uint8)
            rgb = o3d.geometry.Image(rgb_array)
            depth_array = (sample["depth"] / 1000.0).astype(np.float32)
            depth = o3d.geometry.Image(depth_array)

            # Create RGBD image with explicit convert_rgb_to_intensity=False
            rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
                rgb,
                depth,
                depth_scale=1.0,
                depth_trunc=self.depth_max,
                convert_rgb_to_intensity=False,
            )

            intrinsic = o3d.camera.PinholeCameraIntrinsic(
                width=sample["rgb"].shape[1],
                height=sample["rgb"].shape[0],
                fx=self.depth_intrinsics[0, 0],
                fy=self.depth_intrinsics[1, 1],
                cx=self.depth_intrinsics[0, 2],
                cy=self.depth_intrinsics[1, 2],
            )

            volume.integrate(rgbd_image, intrinsic, np.linalg.inv(sample["pose"]))

        return volume

    def get_scene_pcd(self, voxel_size: float = 0.01) -> o3d.geometry.PointCloud:
        """
        Creates a point cloud from the RGB-D images in the dataset.

        Args:
            voxel_size (float): The size of each voxel in meters for downsampling.

        Returns:
            o3d.geometry.PointCloud: The aggregated point cloud.
        """
        scene_pcd = o3d.geometry.PointCloud()
        for i in tqdm.tqdm(range(0, len(self), 7), desc="Creating scene point cloud"):
            sample = self[i]
            rgb_array = sample["rgb"].astype(np.uint8)
            rgb = o3d.geometry.Image(rgb_array)
            depth_array = (sample["depth"] / 1000.0).astype(np.float32)
            depth = o3d.geometry.Image(depth_array)

            # Create RGBD image with explicit convert_rgb_to_intensity=False
            rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
                rgb,
                depth,
                depth_scale=1.0,
                depth_trunc=self.depth_max,
                convert_rgb_to_intensity=False,
            )

            intrinsic = o3d.camera.PinholeCameraIntrinsic(
                width=sample["rgb"].shape[1],
                height=sample["rgb"].shape[0],
                fx=self.depth_intrinsics[0, 0],
                fy=self.depth_intrinsics[1, 1],
                cx=self.depth_intrinsics[0, 2],
                cy=self.depth_intrinsics[1, 2],
            )

            # Create point cloud from RGBD image
            pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, intrinsic)
            # Transform the point cloud to the camera pose
            pcd.transform(sample["pose"])
            # Add the point cloud to the scene
            scene_pcd += pcd
        # Downsample the point cloud
        scene_pcd = scene_pcd.voxel_down_sample(voxel_size=voxel_size)
        return scene_pcd


def vis_cam_poses(dataset):

    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.add_geometry(o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1))

    for i in range(len(dataset)):
        sample = dataset[i]
        pose = sample["pose"]
        sephere = o3d.geometry.TriangleMesh.create_sphere(radius=0.01)
        sephere.compute_vertex_normals()
        sephere.paint_uniform_color([1, 0, 0])
        sephere.transform(pose)
        if i > 0:
            line = o3d.geometry.LineSet()
            line.points = o3d.utility.Vector3dVector([prev_pose[:3, 3], pose[:3, 3]])
            line.lines = o3d.utility.Vector2iVector([[0, 1]])
            line.colors = o3d.utility.Vector3dVector([[0, 1, 0]])
            vis.add_geometry(line)
        vis.add_geometry(sephere)
        prev_pose = pose
        vis.poll_events()
        vis.update_renderer()
        # vis.clear_geometries()
    vis.run()
    vis.destroy_window()


# Example usage:
if __name__ == "__main__":

    root_path = Path("/home/USER/data/arti4d/raw/din080/scene_2025-04-11-12-58-58")
    cfg = {
        "root_dir": "data",
        "transforms": None,
        "depth_min": 0.5,
        "depth_max": 3.0,
        "root_path": root_path,
        "tf_file_path": "calibration/azure_tf.json",
        "flipped": False,
        "gt_poses": False,
        "droid_slam": True,
    }

    dataset = Arti4DDataset(cfg)
    print("Dataset length:", len(dataset))

    scene_pcd = dataset.get_scene_pcd(voxel_size=0.025)
    scene_pcd_gt = o3d.io.read_point_cloud(
        root_path / "compressed_point_cloud.ply"
    ).voxel_down_sample(voxel_size=0.025)

    # paint the point clouds
    scene_pcd.paint_uniform_color([0.5, 0.5, 0.5])
    scene_pcd_gt.paint_uniform_color([1, 0, 0])

    print("Scene point cloud size:", len(scene_pcd.points))
    print("Ground truth point cloud size:", len(scene_pcd_gt.points))

    print(dataset.reg_matrix)

    o3d.visualization.draw_geometries([scene_pcd, scene_pcd_gt])

    # # Aggregate TSDF
    # tsdf_volume = dataset.get_scene_tsdf(
    #     voxel_size=0.01,
    #     truncation_distance=0.015,
    # )
    # # # Extract and visualize the mesh from the TSDF volume
    # mesh = tsdf_volume.extract_triangle_mesh()
    # mesh.compute_vertex_normals()
    # gt = o3d.io.read_triangle_mesh(
    #     root_path / "compressed_mesh.ply"
    # )  # Load the ground truth mesh if available
    # o3d.visualization.draw_geometries([mesh, gt])
    # o3d.io.write_triangle_mesh(
    #     "outputs/raw_tsdf_mesh.ply", mesh, write_ascii=True, compressed=True
    # )
