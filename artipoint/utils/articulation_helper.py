import numpy as np
import gtsam
from scipy.spatial.transform import Rotation
import matplotlib.pyplot as plt
import open3d as o3d
from scipy.linalg import null_space
from scipy.optimize import minimize


def ransac_estimate_transformation(pairs, threshold, max_iterations):
    best_inliers = []
    best_transform = None
    number_of_outliers = 0

    for _ in range(max_iterations):
        # Randomly sample 3 points
        sample_indices = np.random.choice(len(pairs), 3, replace=False)
        sampled_pairs = pairs[sample_indices]

        # Estimate transformation from sampled points
        R, t = compute_svd_transform(sampled_pairs)

        # Apply transformation to all points
        transformed_points = (R @ pairs[:, 0].T).T + t
        distances = np.linalg.norm(transformed_points - pairs[:, 1], axis=1)

        # Count inliers
        inliers = np.where(distances < threshold)[0]

        if len(inliers) > len(best_inliers):
            best_inliers = inliers
            best_transform = (R, t)

        # count number of outliers
        number_of_outliers = max(number_of_outliers, len(pairs) - len(inliers))

    print(f"Number of outliers: {number_of_outliers}")

    # Refine using all inliers
    if best_transform:
        R, t = compute_svd_transform(pairs[best_inliers])
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = t
    return T


def compute_svd_transform(pairs):
    p, q = pairs[:, 0], pairs[:, 1]
    p_mean, q_mean = np.mean(p, axis=0), np.mean(q, axis=0)
    p_centered, q_centered = p - p_mean, q - q_mean

    H = p_centered.T @ q_centered
    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        R[:, -1] *= -1
    t = q_mean - R @ p_mean
    return R, t


def read_point_pairs(filename):
    pairs = np.load(filename)
    return pairs


def calculate_error(T_gt, R, t):
    T_est = np.eye(4)
    T_est[:3, :3] = R
    T_est[:3, 3] = t
    T_gt_inv = np.linalg.inv(T_gt)
    T_diff = T_gt_inv @ T_est
    R_diff = T_diff[:3, :3]
    t_diff = T_diff[:3, 3]
    angle_diff = Rotation.from_matrix(R_diff).magnitude()
    translation_diff = np.linalg.norm(t_diff)
    return angle_diff, translation_diff


def estimate_motion_point_l2(transformations):
    """Find the point on the axis of motion using L2 norm minimization, which is the point that didn't move during rotation


    Args:
        transformations (list): List of SE(3) transformations

    Returns:
        np.ndarray: 3D point that lies on the rotation axis
    """

    rotations = [T[:3, :3] for T in transformations]
    translations = [T[:3, 3] for T in transformations]

    # Method1: Find a point on the rotation axis
    # The rotation axis passes through points that don't move during rotation
    # We can find this by solving: (I - R)p = t for multiple transformations
    # Stack equations from multiple transformations
    A = []
    b = []
    for R, t in zip(rotations[1:], translations[1:]):  # Skip the first one
        A.append(np.eye(3) - R)
        b.append(t)

    A = np.vstack(A)
    b = np.concatenate(b)

    # Solve using least squares to find a point on the axis
    try:
        point, residuals, rank, s = np.linalg.lstsq(A, b, rcond=None)
    except np.linalg.LinAlgError:
        # If singular matrix, use alternative method
        # Find the null space of A and project t onto it
        null_space_basis = null_space(A)
        point = null_space_basis @ null_space_basis.T @ b

    return point


def find_intersection_point(p1, d1, p2, d2):
    """
    Find the intersection point of two lines defined by point and direction
    Using the method from: https://math.stackexchange.com/questions/675203/calculating-centre-of-rotation-given-point-coordinates-at-different-positions
    """
    # Create vectors for computation
    u = d1
    v = d2
    w = p1 - p2

    # Calculate parameters
    a = np.dot(u, u)
    b = np.dot(u, v)
    c = np.dot(v, v)
    d = np.dot(u, w)
    e = np.dot(v, w)

    # Calculate parameters for the intersection point
    denom = a * c - b * b
    if abs(denom) < 1e-10:
        # Lines are nearly parallel
        return (p1 + p2) / 2
    s = (b * e - c * d) / denom
    # Calculate intersection point
    intersection = p1 + s * u

    return intersection


def get_perpendicular_bisector(p1, p2, axis):
    """
    Returns a point on the perpendicular bisector and its direction vector
    """
    # Midpoint
    mid_point = (p1 + p2) / 2

    # Direction vector of p1p2
    direction = p2 - p1

    # Project direction onto plane perpendicular to rotation axis
    # This ensures our perpendicular bisector is parallel to the rotation axis
    direction_proj = direction - np.dot(direction, axis) * axis

    # Get perpendicular direction in the plane
    perp_direction = np.cross(axis, direction_proj)
    if np.linalg.norm(perp_direction) > 0:
        perp_direction = perp_direction / np.linalg.norm(perp_direction)

    return mid_point, perp_direction


def estimate_motion_axis_eigen(transformations):
    """
    Estimate the axis of motion and a point on that axis from a sequence of SE(3) transformations.
    using Eigenvalue decomposition of the combined rotation matrix

    Args:
        transformations: List of 4x4 SE(3) transformation matrices

    Returns:
        axis: 3D unit vector representing the rotation axis
        point: 3D point that lies on the rotation axis
    """
    # Extract rotation matrices and translations
    rotations = [T[:3, :3] for T in transformations]
    translations = [T[:3, 3] for T in transformations]

    # Estimate rotation axis using multiple rotation matrices
    combined_rotations = np.eye(3)
    for R in rotations[1:]:
        combined_rotations = combined_rotations @ R

    # Get eigenvalues and eigenvectors for rotation axis
    eigenvals, eigenvecs = np.linalg.eig(combined_rotations)
    idx = np.argmin(np.abs(eigenvals - 1))
    axis = np.real(eigenvecs[:, idx])
    axis = axis / np.linalg.norm(axis)

    return axis


def estimate_motion_point_bisectors(transformations, axis):

    # Extract translations
    translations = [T[:3, 3] for T in transformations]

    # Method 2: Find intersection of perpendicular bisectors
    # Get perpendicular bisectors for consecutive pairs
    bisector_points = []
    bisector_directions = []

    for i in range(len(translations) - 1):
        mid_point, perp_dir = get_perpendicular_bisector(
            np.array(translations[i]), np.array(translations[i + 1]), axis
        )
        bisector_points.append(mid_point)
        bisector_directions.append(perp_dir)

    # Find intersection points of all pairs of bisectors
    intersection_points = []
    for i in range(len(bisector_points) - 1):
        intersection = find_intersection_point(
            bisector_points[i],
            bisector_directions[i],
            bisector_points[i + 1],
            bisector_directions[i + 1],
        )
        intersection_points.append(intersection)

    # Take the average of all intersection points as our final point
    point = (
        np.mean(intersection_points, axis=0)
        if intersection_points
        else bisector_points[0]
    )
    return point


def point_line_dist(p, x, axis):
    """Computes the distance between a point and a line defined by a supporting point and a direction vector.

    Args:
        p (_type_): Point
        x (_type_): Supporting point
        axis (_type_): Direction of line

    Returns:
        float: float
    """
    return np.linalg.norm(np.cross(p - x, axis), axis=-1)


def estimate_motion_point(point_pairs, axis):
    """Estimate the point on the axis of motion using the point pairs and axis of motion
    min sim_i{|| (p-x) x axis ||^2}, x is the point on the axis of motion

    Args:
        point_pairs (_type_): _description_
        axis (_type_): _description_
    """

    p = point_pairs[:, :, 0].reshape(-1, 3)
    # q = point_pairs[:, :, 1]
    # p = np.stack([p, q], axis=1).reshape(-1, 3)

    loss = lambda x: np.sum(point_line_dist(p, x, axis) ** 2)

    # Find the point on the axis of motion
    # x0 = np.mean(p, axis=0)
    x0 = np.array([1, 1, 0])
    print(f"Initial guess: {x0}")
    print(f"Inital loss: {loss(x0)}")
    res = minimize(loss, x0, method="Nelder-Mead")
    print(f"Optimized point: {res.x}")
    print(f"Optimized loss: {res.fun}")
    return res.x


def estimate_motion_axis_twist(transformations):
    """Find the axis of motion using the twist representation of SE(3) transformations

    Args:
        transformations (List: SE3): List of SE(3) transformations

    Returns:
        np.ndarray: 3D unit vector representing the rotation axis
    """
    # Extract the compined SE3 transformation
    T = transformations[0]
    for T_i in transformations[1:]:
        T = T @ T_i
    twist = gtsam.Pose3.Logmap(gtsam.Pose3(T))
    w, v = twist[:3], twist[3:]
    w_norm = np.linalg.norm(w)
    v_norm = np.linalg.norm(v)
    print(f"Norm of w: {w_norm}")
    print(f"Norm of v: {v_norm}")
    # classify the axis based on the norm of w and v
    eps = 0.3
    if w_norm > eps:
        print("Revolute joint")
        axis = w / w_norm
    else:
        print("Prismatic joint")
        axis = v / v_norm
    return axis


def estimate_motion_point_twist(transformations):
    """Find the point on the axis of motion using the twist representation of SE(3) transformations
    it find the point that lies on the rotation axis and with minimum distance to the origin [0, 0, 0]
    c = (v x w) / ||w||^2

    Args:
        transformations (List: SE3): List of SE(3) transformations

    Returns:
        np.ndarray: 3D point that lies on the rotation axis
    """
    # Extract the compined SE3 transformation
    T = transformations[0]
    for T_i in transformations[1:]:
        T = T @ T_i
    twist = gtsam.Pose3.Logmap(gtsam.Pose3(T))
    w = twist[:3]
    v = twist[3:]
    w_norm_sq = np.dot(w, w)
    point = np.cross(v, w) / w_norm_sq
    return point


def angle_between_vectors(v1, v2):
    """Calculate the angle between two vectors in radians.

    Args:
        v1 (np.ndarray): First input vector
        v2 (np.ndarray): Second input vector

    Returns:
        float: Angle between vectors in radians
    """
    return np.arccos(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))


def compare_SE3(T1, T2):
    """Compare two SE(3) transformations

    Args:
        T1 (np.ndarray): First SE(3) transformation
        T2 (np.ndarray): Second SE(3) transformation

    Returns:
        float: Angle between the two rotations in radians
        float: Translation difference in meters
    """
    T_diff = np.linalg.inv(T1) @ T2
    angle_error = Rotation.from_matrix(T_diff[:3, :3]).magnitude()
    translation_error = np.linalg.norm(T_diff[:3, 3])
    return angle_error, translation_error


def plot_results(
    pairs,
    p_transformeds,
    gt_pairs=None,
    axis=None,
    center=None,
    gt_axis=None,
    gt_center=None,
):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    for i in range(pairs.shape[0]):
        ax.scatter(
            pairs[i, :, 0, 0],
            pairs[i, :, 0, 1],
            pairs[i, :, 0, 2],
            color="blue",
            marker="x",
            alpha=0.5,
        )
        ax.scatter(
            pairs[i, :, 1, 0],
            pairs[i, :, 1, 1],
            pairs[i, :, 1, 2],
            color="blue",
            marker="x",
            alpha=0.5,
        )
    ax.scatter(0, 0, 0, c="b", marker="x", label="Observed")
    for i in range(p_transformeds.shape[0]):
        ax.scatter(
            p_transformeds[i, :, 0],
            p_transformeds[i, :, 1],
            p_transformeds[i, :, 2],
            color="green",
            marker="o",
            alpha=0.75,
        )
    ax.scatter(0, 0, 0, c="g", marker="o", label="Estimated")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    if gt_pairs is not None:
        for i in range(gt_pairs.shape[0]):
            ax.scatter(
                gt_pairs[i, :, 0, 0],
                gt_pairs[i, :, 0, 1],
                gt_pairs[i, :, 0, 2],
                color="red",
                alpha=0.5,
                marker="x",
            )
            ax.scatter(
                gt_pairs[i, :, 1, 0],
                gt_pairs[i, :, 1, 1],
                gt_pairs[i, :, 1, 2],
                color="red",
                alpha=0.5,
                marker="x",
            )
        ax.scatter(0, 0, 0, c="r", marker="x", label="Ground truth")
    # plot axis of motion if provided
    if axis is not None:
        if center is None:
            ax.quiver(
                0,
                0,
                0,
                axis[0],
                axis[1],
                axis[2],
                color="black",
                label="Axis of rotation",
            )
        else:
            ax.quiver(
                center[0],
                center[1],
                center[2],
                axis[0],
                axis[1],
                axis[2],
                color="black",
                label="Axis of rotation",
            )
    # plot center of rotation if provided
    if center is not None:
        ax.scatter(
            center[0],
            center[1],
            center[2],
            c="r",
            marker="o",
            label="Center of rotation",
            s=200,
        )
    # plot gt axis of motion if provided
    if gt_axis is not None:
        if gt_center is not None:
            ax.quiver(
                gt_center[0],
                gt_center[1],
                gt_center[2],
                gt_axis[0],
                gt_axis[1],
                gt_axis[2],
                color="green",
                label="GT Axis of rotation",
            )
        else:
            ax.quiver(
                0,
                0,
                0,
                gt_axis[0],
                gt_axis[1],
                gt_axis[2],
                color="green",
                label="GT Axis of rotation",
            )
    # plot gt center of rotation if provided
    if gt_center is not None:
        ax.scatter(
            gt_center[0],
            gt_center[1],
            gt_center[2],
            c="g",
            marker="o",
            label="GT Center of rotation",
            s=100,
        )

    # set axis limits
    all_points = np.concatenate(
        [
            pairs[:, :, 0, :].reshape(-1, 3),
            pairs[:, :, 1, :].reshape(-1, 3),
            p_transformeds.reshape(-1, 3),
        ]
    )
    max_range = np.array(
        [
            all_points[:, 0].max() - all_points[:, 0].min(),
            all_points[:, 1].max() - all_points[:, 1].min(),
            all_points[:, 2].max() - all_points[:, 2].min(),
        ]
    ).max()
    mid_x = (all_points[:, 0].max() + all_points[:, 0].min()) / 2
    mid_y = (all_points[:, 1].max() + all_points[:, 1].min()) / 2
    mid_z = (all_points[:, 2].max() + all_points[:, 2].min()) / 2
    ax.set_xlim(mid_x - max_range / 2, mid_x + max_range / 2)
    ax.set_ylim(mid_y - max_range / 2, mid_y + max_range / 2)
    ax.set_zlim(mid_z - max_range / 2, mid_z + max_range / 2)

    # add legnds
    plt.legend()
    plt.show()


def relative_to_absolute_transformations(relative_transformations):
    """
    Convert a list of relative transformations to absolute transformations.

    Parameters:
        relative_transformations (list of np.ndarray): List of relative 4x4 transformation matrices.

    Returns:
        list of np.ndarray: List of absolute 4x4 transformation matrices.
    """
    # Initialize the list of absolute transformations
    absolute_transformations = []

    # Start with the first transformation as the identity matrix
    current_absolute = np.eye(4)

    for rel_transform in relative_transformations:
        # Update the absolute transformation by multiplying with the relative one
        current_absolute = np.dot(current_absolute, rel_transform)
        # Append the new absolute transformation
        absolute_transformations.append(current_absolute.copy())

    return absolute_transformations


import open3d as o3d
import numpy as np


def plot_results_o3d(
    pairs,
    p_transformeds,
    gt_pairs=None,
    axis=None,
    center=None,
    gt_axis=None,
    gt_center=None,
):
    # Create a visualization object
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="3D Point Cloud Visualization")

    # Create geometries for the observed points (blue)
    for i in range(pairs.shape[0]):
        observed_points1 = o3d.geometry.PointCloud()
        observed_points1.points = o3d.utility.Vector3dVector(pairs[i, :, 0, :])
        observed_points1.paint_uniform_color([0, 0, 1])  # blue
        vis.add_geometry(observed_points1)

        observed_points2 = o3d.geometry.PointCloud()
        observed_points2.points = o3d.utility.Vector3dVector(pairs[i, :, 1, :])
        observed_points2.paint_uniform_color([0, 0, 1])  # blue
        vis.add_geometry(observed_points2)

    # Add a reference point at the origin (blue)
    origin_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5)
    origin_frame.translate([0, 0, 0])
    vis.add_geometry(origin_frame)

    # Create geometry for transformed points (green)
    for i in range(p_transformeds.shape[0]):
        transformed_pc = o3d.geometry.PointCloud()
        transformed_pc.points = o3d.utility.Vector3dVector(p_transformeds[i])
        transformed_pc.paint_uniform_color([0, 1, 0])  # green
        vis.add_geometry(transformed_pc)

    # Add ground truth pairs if provided (red)
    if gt_pairs is not None:
        for i in range(gt_pairs.shape[0]):
            gt_points1 = o3d.geometry.PointCloud()
            gt_points1.points = o3d.utility.Vector3dVector(gt_pairs[i, :, 0, :])
            gt_points1.paint_uniform_color([1, 0, 0])  # red
            vis.add_geometry(gt_points1)

            gt_points2 = o3d.geometry.PointCloud()
            gt_points2.points = o3d.utility.Vector3dVector(gt_pairs[i, :, 1, :])
            gt_points2.paint_uniform_color([1, 0, 0])  # red
            vis.add_geometry(gt_points2)

    # Add axis of motion if provided (white)
    if axis is not None:
        axis_line = o3d.geometry.LineSet()
        start_point = [0, 0, 0] if center is None else center
        # Scale axis for better visualization
        axis_scaled = axis / np.linalg.norm(axis)
        end_point = [
            start_point[0] + axis_scaled[0],
            start_point[1] + axis_scaled[1],
            start_point[2] + axis_scaled[2],
        ]
        points = [start_point, end_point]
        lines = [[0, 1]]

        axis_line.points = o3d.utility.Vector3dVector(points)
        axis_line.lines = o3d.utility.Vector2iVector(lines)
        axis_line.colors = o3d.utility.Vector3dVector([[1, 1, 1]])  # white
        vis.add_geometry(axis_line)

    # Add center of rotation if provided (red)
    if center is not None:
        center_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.05)
        center_sphere.translate(center)
        center_sphere.paint_uniform_color([1, 0, 0])  # red
        vis.add_geometry(center_sphere)

    # Add ground truth axis if provided (green)
    if gt_axis is not None:
        gt_axis_line = o3d.geometry.LineSet()
        gt_start_point = [0, 0, 0] if gt_center is None else gt_center
        # Scale axis for better visualization
        gt_axis_scaled = gt_axis / np.linalg.norm(gt_axis)
        gt_end_point = [
            gt_start_point[0] + gt_axis_scaled[0],
            gt_start_point[1] + gt_axis_scaled[1],
            gt_start_point[2] + gt_axis_scaled[2],
        ]
        gt_points = [gt_start_point, gt_end_point]
        gt_lines = [[0, 1]]

        gt_axis_line.points = o3d.utility.Vector3dVector(gt_points)
        gt_axis_line.lines = o3d.utility.Vector2iVector(gt_lines)
        gt_axis_line.colors = o3d.utility.Vector3dVector([[0, 1, 0]])  # green
        vis.add_geometry(gt_axis_line)

    # Add ground truth center if provided (green)
    if gt_center is not None:
        gt_center_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.05)
        gt_center_sphere.translate(gt_center)
        gt_center_sphere.paint_uniform_color([0, 1, 0])  # green
        vis.add_geometry(gt_center_sphere)

    # Set rendering options
    render_option = vis.get_render_option()
    render_option.point_size = 5.0
    render_option.background_color = np.array([0, 0, 0])  # white background
    render_option.line_width = 5.0

    # Print legend information since Open3D doesn't have built-in legends
    print("Legend:")
    print("- Blue points: Observed point pairs")
    print("- Green points: Estimated transformed points")
    if gt_pairs is not None:
        print("- Red points: Ground truth point pairs")
    if axis is not None:
        print("- Black line: Estimated axis of rotation")
    if center is not None:
        print("- Red sphere: Estimated center of rotation")
    if gt_axis is not None:
        print("- Green line: Ground truth axis of rotation")
    if gt_center is not None:
        print("- Green sphere: Ground truth center of rotation")

    # Run the visualization
    vis.run()
    vis.destroy_window()
