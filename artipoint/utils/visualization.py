import copy
import os
import sys
import time
import subprocess
from pathlib import Path
from typing import List, Optional, Union

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
import torch
from scipy.spatial.transform import Rotation as R


def create_coordinate_frame(size=1.0):
    """Create a coordinate frame for visualization"""
    return o3d.geometry.TriangleMesh.create_coordinate_frame(size=size)


def tensor_to_o3d_point_cloud(points_tensor):
    """Convert PyTorch tensor to Open3D point cloud"""
    # Move tensor to CPU and convert to numpy
    points_np = points_tensor.detach().cpu().numpy()

    # Create Open3D point cloud object
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_np)
    return pcd


def sample_points_from_mesh(mesh, num_points=1000):
    """
    Sample points from a mesh to convert it into a point cloud-like structure.

    Args:
        mesh: The TriangleMesh object (e.g., a coordinate frame).
        num_points: Number of points to sample from the mesh.

    Returns:
        point_cloud: A PointCloud object containing the sampled points.
    """
    pcd = mesh.sample_points_uniformly(number_of_points=num_points)
    return pcd


def calculate_zy_rotation_for_arrow(vec):
    gamma = np.arctan2(vec[1], vec[0])
    Rz = np.array(
        [
            [np.cos(gamma), -np.sin(gamma), 0],
            [np.sin(gamma), np.cos(gamma), 0],
            [0, 0, 1],
        ]
    )

    vec = Rz.T @ vec

    beta = np.arctan2(vec[0], vec[2])
    Ry = np.array(
        [[np.cos(beta), 0, np.sin(beta)], [0, 1, 0], [-np.sin(beta), 0, np.cos(beta)]]
    )
    return Rz, Ry


def visualize_trajectory(
    free_trajectories=None,
    pcd=None,
    axes=None,
    motion_centers=None,
    cam_poses=None,
    save_frames=False,
    output_dir=None,
    output_prefix="frame_",
    save_video=False,
    fps=15,
    resolution=(1280, 720),
):
    """
    Visualize original and transformed point cloud trajectories

    Args:
        trajectories: List of lists of SE(3) transformation matrices
        axes: List of motion axes
        pcd: Point cloud to visualize
        gt_trajectory: Ground truth trajectory (list of SE(3) matrices)
        free_trajectories: List of lists of free trajectories
        motion_centers: List of motion centers
        cam_poses: List of camera poses for rendering viewpoints
        save_frames: Whether to save rendered frames as images
        output_dir: Directory to save rendered frames (created if it doesn't exist)
        output_prefix: Prefix for output frame filenames
        save_video: Whether to compile frames into a video (requires ffmpeg)
        fps: Frames per second for the video
        resolution: (width, height) tuple for rendering resolution
    """

    # Create output directory if saving frames
    if save_frames and output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        print(f"Frames will be saved to {output_dir}")

    # Create Open3D visualization window with specified resolution
    vis = o3d.visualization.Visualizer()
    vis.create_window(width=resolution[0], height=resolution[1])

    # Set rendering options for better quality
    render_option = vis.get_render_option()
    render_option.point_size = 2.0
    render_option.background_color = np.array([1, 1, 1])  # White background

    # add pcd if provided
    if pcd is not None:
        vis.add_geometry(pcd)

    # Handle multiple trajectories with different colors
    colors = [
        [0, 0, 1],
        [1, 0, 1],
        [0, 1, 1],
        [1, 1, 0],
    ]  # Blue, Magenta, Cyan, Yellow

    # Make free_trajectories a list of lists if it's a single trajectory
    if free_trajectories:
        # add free trajectories
        for traj_idx, free_trajectory in enumerate(free_trajectories):
            for i, transform in enumerate(free_trajectory):
                frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.05)
                frame.transform(transform)
                vis.add_geometry(frame)

    # Default motion center if not provided
    if motion_centers is None and axes is not None:
        motion_centers = [[0, 0, 0]] * len(axes)

    # add axes of motion if provided
    if axes is not None:
        for i, axis in enumerate(axes):
            # Get the center and normalize the axis
            center = np.array(motion_centers[i])
            axis_norm = np.array(axis) / np.linalg.norm(axis)

            # Define arrow properties
            arrow_length = 0.5  # Length of the arrow
            cylinder_radius = 0.01  # Radius of cylinder
            cone_radius = cylinder_radius * 2  # Radius of cone (arrowhead)

            # Create an arrow
            arrow = o3d.geometry.TriangleMesh.create_arrow(
                cylinder_radius=cylinder_radius,
                cone_radius=cone_radius,
                cylinder_height=arrow_length * 0.8,
                cone_height=arrow_length * 0.2,
            )

            # Calculate rotation to align with axis direction
            # Default arrow is along the positive y-axis
            y_axis = np.array([0, 0, 1])

            # Get rotation from y-axis to the axis direction
            rotation_matrix = np.eye(3)
            if not np.allclose(axis_norm, y_axis):
                rotation_axis = np.cross(y_axis, axis_norm)
                if np.linalg.norm(rotation_axis) > 1e-6:  # Check if not zero
                    rotation_axis = rotation_axis / np.linalg.norm(rotation_axis)
                    angle = np.arccos(np.clip(np.dot(y_axis, axis_norm), -1.0, 1.0))
                    rotation_matrix = R.from_rotvec(rotation_axis * angle).as_matrix()

            # Create transformation matrix
            transform = np.eye(4)
            transform[:3, :3] = rotation_matrix
            transform[:3, 3] = center

            # Apply transformation
            arrow.transform(transform)

            # Set color (yellow)
            arrow.paint_uniform_color([1, 1, 0.5])

            # Add to visualizer
            vis.add_geometry(arrow)

    # add motion centers if provided as spheres
    if motion_centers is not None:
        for i, center in enumerate(motion_centers):
            print(f"Motion center {i}: {center}")
            sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.02)
            sphere.translate(center)
            sphere.paint_uniform_color([0, 0, 1])
            vis.add_geometry(arrow)

    # If camera poses are provided and we're saving frames, render from each pose
    frame_files = []
    if cam_poses is not None and save_frames:
        # First render the scene once to initialize
        vis.poll_events()
        vis.update_renderer()

        for i, pose in enumerate(cam_poses):
            # Set camera view based on pose
            param = vis.get_view_control().convert_to_pinhole_camera_parameters()
            # Convert pose to camera parameters
            extrinsic = np.linalg.inv(pose)  # Camera pose is inverse of object pose
            param.extrinsic = extrinsic
            vis.get_view_control().convert_from_pinhole_camera_parameters(param)

            # Update view
            vis.poll_events()
            vis.update_renderer()

            # Save frame
            frame_path = os.path.join(output_dir, f"{output_prefix}{i:04d}.png")
            vis.capture_screen_image(frame_path)
            frame_files.append(frame_path)
            print(f"Saved frame {i+1}/{len(cam_poses)} to {frame_path}")
    else:
        # Run interactive visualization if not saving frames or no camera poses provided
        vis.run()

    vis.destroy_window()

    # Compile video if requested
    if save_video and save_frames and frame_files:
        video_path = os.path.join(output_dir, f"{output_prefix}video.mp4")
        try:
            cmd = [
                "ffmpeg",
                "-y",  # Overwrite output file if it exists
                "-framerate",
                str(fps),
                "-i",
                os.path.join(output_dir, f"{output_prefix}%04d.png"),
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                video_path,
            ]
            subprocess.run(cmd, check=True)
            print(f"Video saved to {video_path}")
        except Exception as e:
            print(f"Failed to create video: {e}")
            print("To manually create video, use ffmpeg:")
            print(
                f"ffmpeg -framerate {fps} -i {output_dir}/{output_prefix}%04d.png -c:v libx264 -pix_fmt yuv420p {video_path}"
            )

    return frame_files if save_frames else None


def create_line_set(points_start, points_end, colors=None):
    """Create LineSet from start and end points."""
    num_points = len(points_start)

    # Create indices for lines
    lines = [[i, i + num_points] for i in range(num_points)]

    # Combine start and end points
    points = np.vstack((points_start, points_end))

    # Create LineSet
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(points)
    line_set.lines = o3d.utility.Vector2iVector(lines)

    if colors is None:
        colors = np.tile(np.array([1, 0, 0]), (len(lines), 1))  # Default red color
    line_set.colors = o3d.utility.Vector3dVector(colors)

    return line_set


def visualize_scene_flow(pc1, pc2, flow=None, voxel_size=None):
    """
    Visualize two point clouds and their scene flow.

    Args:
        pc1 (numpy.ndarray): First point cloud (N x 3)
        pc2 (numpy.ndarray): Second point cloud (N x 3)
        flow (numpy.ndarray, optional): Scene flow vectors (N x 3)
        sample_ratio (float): Ratio of points to visualize (0.0 to 1.0)
    """
    # Create Open3D geometries
    pcd1 = o3d.geometry.PointCloud()
    pcd2 = o3d.geometry.PointCloud()
    # Set points
    pcd1.points = o3d.utility.Vector3dVector(pc1)
    pcd2.points = o3d.utility.Vector3dVector(pc2)
    # Sample points if ratio < 1.0
    if voxel_size is not None:
        pcd1 = pcd1.voxel_down_sample(voxel_size=voxel_size)
        pcd2 = pcd2.voxel_down_sample(voxel_size=voxel_size)

    # Set colors
    pcd1.paint_uniform_color([1, 0, 0])  # Red for first point cloud
    pcd2.paint_uniform_color([0, 0, 1])  # Blue for second point cloud

    # Create visualization list
    vis_list = [pcd1, pcd2]

    # Add flow vectors if provided
    if flow is not None:
        # Create flow line set
        flow_lines = create_line_set(
            pc1,
            pc1 + flow,
            colors=np.tile(
                np.array([0, 1, 0]), (len(pc1), 1)
            ),  # Green for flow vectors
        )
        vis_list.append(flow_lines)

    # Calculate coordinate frame size based on point cloud bounds
    pc_range = np.ptp(pc1, axis=0)  # Gets range in each dimension
    coord_frame_size = np.mean(pc_range) * 0.2  # Use mean of ranges
    # Coordinate frame
    coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=float(coord_frame_size)
    )  # Ensure it's a float
    vis_list.append(coord_frame)

    # Visualize
    o3d.visualization.draw_geometries(
        vis_list,
        window_name="Scene Flow Visualization",
        width=1024,
        height=768,
        left=50,
        top=50,
        point_show_normal=False,
        mesh_show_wireframe=False,
        mesh_show_back_face=False,
    )


# Example usage with the trajectory optimizer
def main():

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create sample point clouds
    num_points = 1000
    num_frames = 5

    # Generate sample point clouds with some random transformation
    point_clouds = []
    base_pc = torch.randn(num_points, 3, device=device)

    for i in range(num_frames):
        translation = torch.tensor([0.1, 0.2, 0.3], device=device) * i
        noise = torch.randn_like(base_pc) * 0.01
        pc = base_pc + translation + noise
        point_clouds.append(pc)

    # Create optimizer and run optimization
    optimizer = TrajectoryOptimizer(point_clouds, device=device)
    optimizer.optimize(num_iterations=1000)

    # Get transformed trajectory
    transformed_trajectory = optimizer.get_transformed_trajectory()

    # Visualize the results
    print("Showing static visualization...")
    visualize_trajectory(point_clouds, transformed_trajectory)

    print("Showing animated visualization...")
    visualize_trajectory_animation(point_clouds, transformed_trajectory)


class TransformVisualizer:
    def __init__(self):
        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window()

        # Set rendering options
        opt = self.vis.get_render_option()
        opt.background_color = np.asarray([0.1, 0.1, 0.1])  # Dark background
        opt.point_size = 2.0

    @staticmethod
    def create_coordinate_frame(
        size: float = 1.0, transform: Optional[np.ndarray] = None
    ) -> o3d.geometry.TriangleMesh:
        """
        Create a coordinate frame with optional transform

        Args:
            size: Size of coordinate frame
            transform: 4x4 homogeneous transformation matrix
        """
        frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=size)

        if transform is not None:
            frame.transform(transform)

        return frame

    @staticmethod
    def create_trajectory_line(
        transforms: List[np.ndarray], color: List[float] = [1, 0, 0]
    ) -> o3d.geometry.LineSet:
        """
        Create a line set connecting the origins of transforms

        Args:
            transforms: List of 4x4 transformation matrices
            color: RGB color for the trajectory line
        """
        points = []
        lines = []

        # Extract translation components
        for transform in transforms:
            points.append(transform[:3, 3])

        # Create lines between consecutive points
        for i in range(len(points) - 1):
            lines.append([i, i + 1])

        # Create line set
        line_set = o3d.geometry.LineSet()
        line_set.points = o3d.utility.Vector3dVector(np.array(points))
        line_set.lines = o3d.utility.Vector2iVector(np.array(lines))

        # Set color
        colors = [color for _ in range(len(lines))]
        line_set.colors = o3d.utility.Vector3dVector(np.array(colors))

        return line_set

    def plot_transform(
        self,
        transform: Union[np.ndarray, torch.Tensor],
        frame_size: float = 1.0,
        show_grid: bool = True,
    ) -> None:
        """
        Plot a single homogeneous transformation

        Args:
            transform: 4x4 homogeneous transformation matrix
            frame_size: Size of coordinate frames
            show_grid: Whether to show reference grid
        """
        # Convert torch tensor to numpy if needed
        if isinstance(transform, torch.Tensor):
            transform = transform.detach().cpu().numpy()

        # Add world coordinate frame
        world_frame = self.create_coordinate_frame(size=frame_size)
        self.vis.add_geometry(world_frame)

        # Add transformed coordinate frame
        transformed_frame = self.create_coordinate_frame(
            size=frame_size, transform=transform
        )
        self.vis.add_geometry(transformed_frame)

        # Add reference grid
        if show_grid:
            grid = o3d.geometry.TriangleMesh.create_box(
                width=0.05, height=0.05, depth=0.001
            )
            grid.compute_vertex_normals()
            grid.paint_uniform_color([0.5, 0.5, 0.5])
            for i in range(-5, 6, 1):
                for j in range(-5, 6, 1):
                    grid_copy = copy.deepcopy(grid)
                    grid_copy.translate(np.array([i, j, -0.5]))
                    self.vis.add_geometry(grid_copy)

        # Set default viewpoint
        ctr = self.vis.get_view_control()
        ctr.set_zoom(0.8)
        ctr.set_front([0.5, 0.5, -0.5])
        ctr.set_up([0.0, 1.0, 0.0])

    def plot_transform_sequence(
        self,
        transforms: List[Union[np.ndarray, torch.Tensor]],
        frame_size: float = 1.0,
        show_trajectory: bool = True,
        show_grid: bool = True,
    ) -> None:
        """
        Plot a sequence of homogeneous transformations

        Args:
            transforms: List of 4x4 homogeneous transformation matrices
            frame_size: Size of coordinate frames
            show_trajectory: Whether to show trajectory line
            show_grid: Whether to show reference grid
        """
        # Convert torch tensors to numpy if needed
        transforms_np = []
        for transform in transforms:
            if isinstance(transform, torch.Tensor):
                transforms_np.append(transform.detach().cpu().numpy())
            else:
                transforms_np.append(transform)

        # Add world coordinate frame
        world_frame = self.create_coordinate_frame(size=frame_size)
        self.vis.add_geometry(world_frame)

        # Add transformed coordinate frames
        for transform in transforms_np:
            frame = self.create_coordinate_frame(size=frame_size, transform=transform)
            self.vis.add_geometry(frame)

        # Add trajectory line
        if show_trajectory:
            trajectory = self.create_trajectory_line(transforms_np)
            self.vis.add_geometry(trajectory)

        # Add reference grid
        if show_grid:
            grid = o3d.geometry.TriangleMesh.create_box(
                width=0.05, height=0.05, depth=0.001
            )
            grid.compute_vertex_normals()
            grid.paint_uniform_color([0.5, 0.5, 0.5])
            for i in range(-5, 6, 1):
                for j in range(-5, 6, 1):
                    grid_copy = copy.deepcopy(grid)
                    grid_copy.translate(np.array([i, j, -0.5]))
                    self.vis.add_geometry(grid_copy)

        # Set default viewpoint
        ctr = self.vis.get_view_control()
        ctr.set_zoom(0.8)
        ctr.set_front([0.5, 0.5, -0.5])
        ctr.set_up([0.0, 1.0, 0.0])

    def show(self):
        """Run the visualizer"""
        self.vis.run()
        self.vis.destroy_window()


def visualize_pairs_with_motion_axis(
    pairs, pairs_est, axis, motion_center, point_cloud=None
):
    """
    Visualize the pairs of 3D points along with the estimated axis of motion and center of motion.

    Args:
        pairs (list): List of pairs of 3D points for each frame.
        pairs_est (list): List of estimated pairs of 3D points for each frame.
        axis (np.ndarray): Estimated axis of motion.
        motion_center (np.ndarray): Estimated center of motion.
        point_cloud (np.ndarray, optional): Point cloud data to be plotted. Defaults to None.
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    # Plot the original pairs
    for pair in pairs:
        ax.plot(
            pair[:, 0, 0],
            pair[:, 0, 1],
            pair[:, 0, 2],
            "bo-",
            label="Original Pair" if pair is pairs[0] else "",
        )
        ax.plot(
            pair[:, 1, 0],
            pair[:, 1, 1],
            pair[:, 1, 2],
            "ro-",
            label="Original Pair" if pair is pairs[0] else "",
        )

    # Plot the estimated pairs
    for pair_est in pairs_est:
        ax.plot(
            pair_est[:, 0],
            pair_est[:, 1],
            pair_est[:, 2],
            "go-",
            label="Estimated Pair" if pair_est is pairs_est[0] else "",
        )

    # Plot the axis of motion
    axis_line = np.array([motion_center - axis * 1.5, motion_center + axis * 1.5])
    ax.plot(
        axis_line[:, 0], axis_line[:, 1], axis_line[:, 2], "k-", label="Axis of Motion"
    )

    # Plot the center of motion
    ax.scatter(
        motion_center[0],
        motion_center[1],
        motion_center[2],
        c="k",
        marker="x",
        s=100,
        label="Center of Motion",
    )

    # Plot the point cloud if provided
    if point_cloud is not None:
        ax.scatter(
            point_cloud[:, 0],
            point_cloud[:, 1],
            point_cloud[:, 2],
            c="gray",
            marker=".",
            s=1,
            label="Point Cloud",
        )

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.legend()
    plt.show()


def plot_3d_tracks(
    points,
    visibles,
    rgb_images,
    depth_images,
    infront_cameras=None,
    tracks_leave_trace=16,
    show_occ=False,
    fps=15,
    azure_dataset=None,
    clusters_labels_segments=None,
    camera_poses=None,
    save_frames=False,
    output_dir=None,
    output_prefix="frame_",
    save_video=False,
    resolution=(1280, 720),
):
    """
    Visualize 3D point trajectories frame-by-frame using Open3D.
    Args:
        points (np.ndarray): 3D points of shape (num_frames, num_points, 3) it should be in Global coordinate system.
        visibles (np.ndarray): Visibility of points of shape (num_frames, num_points).
        rgb_images (np.ndarray): RGB images of shape (num_frames, H, W, 3).
        depth_images (np.ndarray): Depth images of shape (num_frames, H, W).
        infront_cameras (np.ndarray): Visibility of points in front of cameras of shape (num_frames, num_points).
        tracks_leave_trace (int): Number of frames to leave trace for each point.
        show_occ (bool): Show occlusion if True, else show visibility.
        fps (int): Frames per second for visualization.
        azure_dataset (AzureProcessor): Azure dataset processor.
        clusters_labels_segments (np.ndarray): Cluster labels for each point.
        camera_poses (np.ndarray): Camera poses for each frame.
        save_frames (bool): Whether to save rendered frames as images.
        output_dir (str): Directory to save rendered frames. Created if it doesn't exist.
        output_prefix (str): Prefix for output frame filenames.
        save_video (bool): Whether to compile frames into a video (requires ffmpeg).
        resolution (tuple): Width and height resolution for rendering window.
    """

    num_frames, num_points = points.shape[0:2]

    # Create distinct colors for different clusters
    color_map = cm.get_cmap("tab10")  # Using tab10 colormap for distinct cluster colors

    # Default colors if no clusters provided
    if clusters_labels_segments is None:
        colors = [color_map(i % 10)[:3] for i in range(num_points)]
    else:
        # Get unique cluster IDs
        unique_clusters = np.unique(clusters_labels_segments)
        cluster_colors = {
            c: color_map(i % 10)[:3] for i, c in enumerate(unique_clusters)
        }
        # Assign colors based on cluster labels
        colors = [
            cluster_colors[clusters_labels_segments[i]] for i in range(num_points)
        ]

    if infront_cameras is None:
        infront_cameras = np.ones_like(visibles).astype(bool)

    # Create output directory if saving frames
    if save_frames and output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        print(f"Frames will be saved to {output_dir}")

    # Setup visualizer with specified resolution
    vis = o3d.visualization.Visualizer()
    vis.create_window(width=resolution[0], height=resolution[1])

    # Set rendering options for better quality output
    render_option = vis.get_render_option()
    render_option.point_size = 2.0
    render_option.background_color = np.array([1, 1, 1])  # White background

    # Keep track of geometries and camera parameters
    camera_params = None  # To store the camera viewpoint
    scene_pc = o3d.geometry.PointCloud()
    frame_files = []

    # add scene pcd as initial frame
    if azure_dataset is not None:
        pcd, _ = azure_dataset.create_pcd(
            rgb_images[0], depth_images[0], camera_poses[0]
        )
        scene_pc += pcd

    for t in range(num_frames):
        if t > 0:
            vis.clear_geometries()  # Clear previous frame's geometries

        vis.add_geometry(scene_pc)  # Add the scene point cloud

        # add the scene point cloud to the visualizer
        if azure_dataset is not None:
            pcd, _ = azure_dataset.create_pcd(
                rgb_images[t], depth_images[t], camera_poses[t]
            )
            # scene_pc += pcd
            # scene_pc = scene_pc.voxel_down_sample(voxel_size=0.03)
            vis.add_geometry(pcd)

        for i in range(num_points):
            if show_occ:
                visible = infront_cameras[t, i]
            else:
                visible = visibles[t, i]

            if visible:
                # Create sphere for current position
                color = colors[i]
                sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.01)
                sphere.translate(points[t, i])
                sphere.paint_uniform_color(color)
                vis.add_geometry(sphere)

                # Trace the trajectory
                trace_start = max(0, t - tracks_leave_trace)
                trace = points[trace_start : t + 1, i]
                trace_vis = visibles[trace_start : t + 1, i]
                trace = trace[trace_vis]
                if len(trace) > 1:
                    line_set = o3d.geometry.LineSet()
                    line_set.points = o3d.utility.Vector3dVector(trace)
                    lines = [[k, k + 1] for k in range(len(trace) - 1)]
                    line_set.lines = o3d.utility.Vector2iVector(lines)
                    line_set.colors = o3d.utility.Vector3dVector([color] * len(lines))
                    vis.add_geometry(line_set)

        if camera_params is not None:
            vis.get_view_control().convert_from_pinhole_camera_parameters(camera_params)

        vis.poll_events()
        vis.update_renderer()
        vis.run()

        # Save frame if requested
        if save_frames and output_dir is not None:
            frame_path = os.path.join(output_dir, f"{output_prefix}{t:04d}.png")
            vis.capture_screen_image(frame_path)
            frame_files.append(frame_path)
            print(f"Saved frame {t}/{num_frames} to {frame_path}")

        # Save the current camera parameters for the next frame
        camera_params = vis.get_view_control().convert_to_pinhole_camera_parameters()

    vis.destroy_window()

    # Compile video if requested
    if save_video and save_frames and frame_files:
        video_path = os.path.join(output_dir, f"{output_prefix}video.mp4")
        try:
            cmd = [
                "ffmpeg",
                "-y",
                "-framerate",
                str(fps),
                "-i",
                os.path.join(output_dir, f"{output_prefix}%04d.png"),
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                video_path,
            ]
            subprocess.run(cmd, check=True)
            print(f"Video saved to {video_path}")
        except Exception as e:
            print(f"Failed to create video: {e}")
            print("To manually create video, use ffmpeg:")
            print(
                f"ffmpeg -framerate {fps} -i {output_dir}/{output_prefix}%04d.png -c:v libx264 -pix_fmt yuv420p {video_path}"
            )

    return frame_files if save_frames else None


def visualize_camera_movement(
    scene_pcd,
    camera_pose_list,
    keypoints=None,
    frame_size=0.2,
    trajectory_color=[0.0, 1.0, 0.0],
    keypoint_color=[1.0, 0.0, 0.0],
    window_width=1280,
    window_height=720,
):
    """
    Visualize a scene point cloud with camera poses, trajectory, and optional keypoints.

    Parameters:
      scene_pcd (o3d.geometry.PointCloud): The scene point cloud.
      camera_pose_list (list of np.ndarray): List of 4x4 camera poses.
      keypoints (np.ndarray, optional): Array of shape (N, 3) or list of arrays with keypoint locations.
                                       If None, no keypoints are visualized.
      animation_speed (float): Controls the speed of the camera animation (lower is slower).
      keypoint_radius (float): Radius of the spheres representing keypoints.
      frame_size (float): Size of the coordinate frames representing camera poses.
      trajectory_color (list): RGB color for the camera trajectory line.
      keypoint_color (list): RGB color for the keypoint spheres.
      save_images (bool): If True, save images of each frame during animation.
      output_dir (str, optional): Directory to save images. Required if save_images is True.
      window_width (int): Width of the visualization window.
      window_height (int): Height of the visualization window.

    Returns:
      None
    """

    # Validate inputs
    if len(camera_pose_list) == 0:
        raise ValueError("camera_pose_list must contain at least one pose")

    # Create a visualizer window with specified dimensions
    vis = o3d.visualization.Visualizer()
    vis.create_window(
        window_name="Camera Trajectory Visualization",
        width=window_width,
        height=window_height,
    )

    # Add the scene point cloud
    vis.add_geometry(scene_pcd)

    # Set rendering options for better visualization
    opt = vis.get_render_option()
    opt.background_color = np.array([0.1, 0.1, 0.1])  # Dark background
    opt.point_size = 1.0

    # create point cloud for keypoints
    if keypoints is not None:
        for i, points in enumerate(keypoints):
            points = (
                camera_pose_list[i][:3, :3] @ points.T
                + camera_pose_list[i][:3, 3][:, None]
            )
            keypoint_pcd = o3d.geometry.PointCloud()
            keypoint_pcd.points = o3d.utility.Vector3dVector(points.T)
            keypoint_pcd.paint_uniform_color(keypoint_color)
            vis.add_geometry(keypoint_pcd)

    # List to store camera positions for trajectory
    camera_positions = []

    # For each camera pose, create a coordinate frame and add it
    for i, pose in enumerate(camera_pose_list):
        # Create a coordinate frame
        frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=frame_size)
        frame.transform(pose)
        vis.add_geometry(frame)

        # Extract camera position from the pose
        camera_positions.append(pose[:3, 3])

    # Create camera trajectory if we have more than one pose
    if len(camera_positions) > 1:
        lines = [[i, i + 1] for i in range(len(camera_positions) - 1)]
        colors = [trajectory_color for _ in lines]

        trajectory = o3d.geometry.LineSet(
            points=o3d.utility.Vector3dVector(camera_positions),
            lines=o3d.utility.Vector2iVector(lines),
        )
        trajectory.colors = o3d.utility.Vector3dVector(colors)
        vis.add_geometry(trajectory)

    # Get the view control
    view_ctl = vis.get_view_control()

    # Set initial camera view
    view_ctl.set_zoom(0.8)

    # First render the scene with all geometries
    vis.poll_events()
    vis.update_renderer()

    # Keep the window open until closed by user
    print("Visualization complete. Close the window to exit.")
    vis.run()
    vis.destroy_window()


def visualize_articulations(
    scene_path,
    gt_data=None,
    predictions=None,
    gt_idcs=None,
    pred_idcs=None,
    axis_length=1.0,
    gt_color=(0, 1, 0),
    pred_color=(1, 0, 0),
):
    """
    Visualize scene mesh/pointcloud and articulation axes (both ground truth and predicted).

    Args:
        scene_path: Path to scene mesh (.obj, .ply) or pointcloud (.pcd, .ply)
        gt_data: Ground truth articulation data from load_gt_data()
        predictions: Predicted articulation data from load_prediction_data() or load_sturm_prediction_data()
        gt_indices: Indices of ground truth that match the predicted data
        pred_indices: Indices of predicted data that match the ground truth
        axis_length: Length of drawn axis lines
        gt_color: RGB color tuple for ground truth axes (default: green)
        pred_color: RGB color tuple for predicted axes (default: red)
    """
    # Create visualization window
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="Articulation Visualization", width=1280, height=720)

    # Load scene geometry
    scene_geometry = None
    file_ext = Path(scene_path).suffix.lower()

    if file_ext in [".obj", ".ply", ".stl"]:
        try:
            scene_geometry = o3d.io.read_triangle_mesh(scene_path)
            scene_geometry.compute_vertex_normals()
            # Add to visualizer with default color
            vis.add_geometry(scene_geometry)
        except Exception as e:
            print(f"Failed to load mesh {e}")
            scene_geometry = None
            return

    # Helper function to create axis line
    def create_axis_line(position, axis_dir, length, color):
        """Create a line segment representing an axis"""
        position = np.array(position)
        axis_dir = np.array(axis_dir)
        axis_dir = axis_dir / np.linalg.norm(axis_dir)  # normalize

        start_point = position - (axis_dir * length / 2)
        end_point = position + (axis_dir * length / 2)

        points = [start_point, end_point]
        lines = [[0, 1]]

        line_set = o3d.geometry.LineSet()
        line_set.points = o3d.utility.Vector3dVector(points)
        line_set.lines = o3d.utility.Vector2iVector(lines)
        line_set.colors = o3d.utility.Vector3dVector([color])

        return line_set

    # Add GT axes
    if gt_data:
        for obj_name, obj_data in gt_data.items():
            if "position" in obj_data and "axis" in obj_data:
                axis_line = create_axis_line(
                    obj_data["position"], obj_data["axis"], axis_length, gt_color
                )
                vis.add_geometry(axis_line)

                # Add a small sphere at the position
                sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.01)
                sphere.translate(obj_data["position"])
                sphere.paint_uniform_color(gt_color)
                vis.add_geometry(sphere)

    # Add prediction axes
    if predictions:
        for pred in predictions:
            if "position" in pred and "axis" in pred:
                axis_line = create_axis_line(
                    pred["position"], pred["axis"], axis_length, pred_color
                )
                vis.add_geometry(axis_line)

                # Add a small sphere at the position
                sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.01)
                sphere.translate(pred["position"])
                print(f"Pred position: {pred['position']}")
                sphere.paint_uniform_color(pred_color)
                vis.add_geometry(sphere)

    # draw line between matched gt and pred indicies
    if gt_idcs is not None and pred_idcs is not None:
        for gt_idx, pred_idx in zip(gt_idcs, pred_idcs):
            if gt_data and predictions:
                gt_obj = gt_data[list(gt_data.keys())[gt_idx]]

                # get gt_data at index gt_idx
                pred_obj = predictions[pred_idx]

                if "position" in gt_obj and "position" in pred_obj:
                    start_point = np.array(gt_obj["position"])
                    end_point = np.array(pred_obj["position"])

                    points = [start_point, end_point]
                    lines = [[0, 1]]
                    line_set = o3d.geometry.LineSet()
                    line_set.points = o3d.utility.Vector3dVector(points)
                    line_set.lines = o3d.utility.Vector2iVector(lines)
                    line_set.colors = o3d.utility.Vector3dVector([(0, 0, 1)])
                    vis.add_geometry(line_set)

                    # plot coordinate frame of the prediction
                    if "w_T_a" in pred_obj.keys():
                        frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
                            size=0.05
                        )
                        frame.transform(pred_obj["w_T_a"])
                        vis.add_geometry(frame)

                        line_set = o3d.geometry.LineSet()
                        line_set.points = o3d.utility.Vector3dVector(
                            pred_obj["motion_paths"]
                        )
                        line_set.lines = o3d.utility.Vector2iVector(
                            [
                                [i, i + 1]
                                for i in range(len(pred_obj["motion_paths"]) - 1)
                            ]
                        )
                        line_set.colors = o3d.utility.Vector3dVector([(1, 1, 0)])
                        vis.add_geometry(line_set)

                        for idx, point in enumerate(pred_obj["motion_paths"]):
                            if idx > 0:
                                line = o3d.geometry.LineSet()
                                line.points = o3d.utility.Vector3dVector(
                                    [prev_point, point]
                                )
                                line.lines = o3d.utility.Vector2iVector([[0, 1]])
                                line.colors = o3d.utility.Vector3dVector([[1, 1, 0]])
                                vis.add_geometry(line)
                            prev_point = point

                        start_point = pred_obj["w_T_a"][:3, 3]
                        end_point = np.array(pred_obj["position"])

                    points = [start_point, end_point]
                    lines = [[0, 1]]
                    line_set = o3d.geometry.LineSet()
                    line_set.points = o3d.utility.Vector3dVector(points)
                    line_set.lines = o3d.utility.Vector2iVector(lines)
                    line_set.colors = o3d.utility.Vector3dVector([(1, 1, 0)])

                    vis.add_geometry(line_set)

    # Set initial camera view
    opt = vis.get_render_option()
    opt.background_color = np.array([0.1, 0.1, 0.1])  # Dark gray background
    opt.point_size = 3.0

    # Run visualization
    vis.run()
    vis.destroy_window()


if __name__ == "__main__":
    import time

    main()
