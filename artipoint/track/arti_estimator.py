import os
import sys
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import loguru
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve

from ultralytics import YOLO

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from cotracker.predictor import CoTrackerPredictor
from sklearn.cluster import DBSCAN


class ArtiEstimator:
    def __init__(self, cfg: Dict[str, Any]) -> None:
        self.device = cfg["device"]
        # Offline cotracker
        if not cfg["cotracker2"]:
            self.cotracker = CoTrackerPredictor(
                checkpoint=os.path.join(cfg["model_path"]),
            ).to(self.device)
            loguru.logger.info(f"Using CoTracker3")
        else:
            self.cotracker = torch.hub.load(
                "facebookresearch/co-tracker", "cotracker2"
            ).to(self.device)
            loguru.logger.info(f"Using CoTracker2")
        self.arti4d_dataset = cfg["arti4d_dataset"]
        self.yolo = YOLO(cfg["yolo_path"])

    @staticmethod
    def _create_queries_bbox(
        bbox: np.ndarray, grid: int = 10, frames: List[int] = [0], device: str = "cuda"
    ) -> torch.Tensor:
        """
        Create a grid of queries around a bounding box.
        """
        x1, y1, x2, y2 = bbox
        x = np.linspace(x1, x2, grid)
        y = np.linspace(y1, y2, grid)
        x, y = np.meshgrid(x, y)
        frames = np.repeat(frames, grid**2)
        queries = torch.tensor(np.stack([frames, x.ravel(), y.ravel()], axis=1)).to(
            device
        )
        return queries.float()

    @staticmethod
    def _create_queries_points(
        points: List[Tuple[int, int]], frames: List[int] = [0], device: str = "cuda"
    ) -> torch.Tensor:
        """
        Create queries from a list of point coordinates.
        """
        points_arr = np.array(points)
        frames_arr = np.repeat(frames, points_arr.shape[0])
        queries = torch.tensor(
            np.concatenate([frames_arr[:, None], points_arr], axis=1)
        ).to(device)
        return queries.float()

    @staticmethod
    def calculate_variance(points: np.ndarray, visibility: np.ndarray) -> np.ndarray:
        """
        Calculate variance for point tracks.
        """
        variances = []
        for track_idx in range(points.shape[1]):
            visible_points = points[visibility[:, track_idx], track_idx]
            if visible_points.shape[0] > 0:
                variances.append(np.var(visible_points, axis=0))
            else:
                variances.append(np.zeros(points.shape[2]))
        return np.array(variances)

    def _cotracker_process(
        self,
        window_frames: List[np.ndarray],
        queries: torch.Tensor,
        backward_tracking: bool = False,
    ) -> torch.Tensor:
        """
        Process a window of frames through the CoTracker.
        """
        video = (
            torch.tensor(np.stack(window_frames))
            .permute(0, 3, 1, 2)[None]
            .float()
            .to(self.device)
        )
        return self.cotracker(
            video, queries=queries[None], backward_tracking=backward_tracking
        )

    @staticmethod
    def _select_points(img: np.ndarray) -> List[Tuple[int, int]]:
        """
        Let the user select points interactively on an image.
        """
        selected_points: List[Tuple[int, int]] = []

        def onclick(event):
            if event.xdata is not None and event.ydata is not None:
                x, y = int(event.xdata), int(event.ydata)
                selected_points.append((x, y))
                plt.scatter(x, y, c="red", s=40)
                plt.draw()

        fig, ax = plt.subplots()
        ax.imshow(img)
        ax.set_title("Click to select points. Press Enter to finish.")
        fig.canvas.mpl_connect("button_press_event", onclick)

        def on_key(event):
            if event.key == "enter":
                plt.close()

        fig.canvas.mpl_connect("key_press_event", on_key)
        plt.show()
        return selected_points

    def _segment_human(self, img: np.ndarray) -> List[np.ndarray]:
        """
        Segment human regions using a YOLO model.
        """
        results = self.yolo.predict(source=img, verbose=False, device=self.device)
        indices = [i for i, cls in enumerate(results[0].boxes.cls) if cls == 0]
        masks = results[0].masks
        human_masks = [masks[i].cpu().numpy().data.transpose(1, 2, 0) for i in indices]
        return human_masks

    def _project_2d_tracks_to_3d(
        self,
        depth_window_frames: List[np.ndarray],
        pred_tracks: np.ndarray,
        min_depth: float = 0.3,
        max_depth: float = 5.0,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Projects 2D tracks to 3D coordinates using depth data.
        """
        pred_tracks_3d = []
        valid_depth_masks = []
        K = np.array(self.arti4d_dataset.depth_intrinsics)
        for i, depth in enumerate(depth_window_frames):
            coord_x = np.clip(pred_tracks[i, :, 0], 0, depth.shape[1] - 1)
            coord_y = np.clip(pred_tracks[i, :, 1], 0, depth.shape[0] - 1)
            p2 = np.stack([coord_x, coord_y, np.ones_like(coord_x)], axis=1)
            d = depth[coord_y.astype(int), coord_x.astype(int)] / 1000.0
            points = np.linalg.inv(K) @ p2.T * d
            valid_mask = np.logical_and(
                np.logical_and(points[2] > min_depth, points[2] < max_depth),
                np.isfinite(points[2]),
            )
            valid_depth_masks.append(valid_mask)
            pred_tracks_3d.append(points.T)
        return np.stack(pred_tracks_3d), np.stack(valid_depth_masks)

    @staticmethod
    def create_points_pairs(
        pred_3d_tracker: np.ndarray,
        pred_visibility: np.ndarray,
        path_frac: float = 0.03,
    ) -> List[np.ndarray]:
        """Create pairs of 3D points based on visibility.

        Args:
            pred_3d_tracker (np.ndarray): Point trajectories (T, F, 3).
            pred_visibility (np.ndarray): Visibility mask (T, F).

        Returns:
            List[np.ndarray]: _description_
        """
        pairs = []
        for i in range(pred_3d_tracker.shape[0]):
            visible_idx = np.where(pred_visibility[i])[0]
            for j in range(i + 1, pred_3d_tracker.shape[0]):
                visible_next_idx = np.where(pred_visibility[j])[0]
                common = np.intersect1d(visible_idx, visible_next_idx)
                if len(common) > 0:
                    pairs.append(
                        np.stack(
                            (pred_3d_tracker[i, common], pred_3d_tracker[j, common]),
                            axis=1,
                        )
                    )
                    break
        return pairs

    @staticmethod
    def select_bbox(image: np.ndarray) -> Tuple[int, int, int, int]:
        """
        Open an interactive window to select a bounding box.
        """
        cv2.namedWindow("Select BBox", cv2.WINDOW_NORMAL)
        roi = cv2.selectROI("Select BBox", image, fromCenter=False, showCrosshair=True)
        cv2.destroyWindow("Select BBox")
        # Convert (x, y, w, h) to (x1, y1, x2, y2)
        return (roi[0], roi[1], roi[0] + roi[2], roi[1] + roi[3])

    def median_filter(
        self,
        pred_3d_tracker: np.ndarray,
        pred_visibility: np.ndarray,
        variance_type: str = "2d",
        percentile: int = 60,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply a median filter to remove static 3D points.
        """
        if variance_type == "2d":
            var = self.calculate_variance(pred_3d_tracker[:, :, :2], pred_visibility)
        elif variance_type == "3d":
            var = self.calculate_variance(pred_3d_tracker, pred_visibility)
        else:
            raise ValueError("Invalid variance_type. Choose either '2d' or '3d'.")
        points_motion = np.linalg.norm(var, axis=1)
        static_points = points_motion < np.percentile(points_motion, percentile)
        loguru.logger.info(
            f"Static points: {np.sum(static_points)}; Dynamic points: {np.sum(~static_points)}"
        )
        pred_3d_tracker_filtered = pred_3d_tracker[:, ~static_points]
        pred_visibility_filtered = pred_visibility[:, ~static_points]
        return pred_3d_tracker_filtered, pred_visibility_filtered

    @staticmethod
    def filter_non_smooth_tracks(
        pred_3d_tracker: np.ndarray, pred_visibility: np.ndarray, thrsh: float = 0.1
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Filter tracks with significant motion change.
        """
        change = np.zeros(pred_3d_tracker.shape[1])
        for i in range(pred_3d_tracker.shape[1]):
            visible_points = pred_3d_tracker[pred_visibility[:, i], i]
            change[i] = (
                np.linalg.norm(np.abs(np.diff(visible_points, axis=0)), axis=0).mean()
                if len(visible_points) > 1
                else 0
            )
        significant_change = change > thrsh
        loguru.logger.info(
            f"Points with significant change: {np.sum(significant_change)}"
        )
        pred_3d_tracker_filtered = pred_3d_tracker[:, ~significant_change]
        pred_visibility_filtered = pred_visibility[:, ~significant_change]
        return pred_3d_tracker_filtered, pred_visibility_filtered

    @staticmethod
    def extract_orb_features(
        image: np.ndarray, max_features: int = 500, grid_size: Tuple[int, int] = None
    ) -> List[Tuple[int, int]]:
        """
        Extract ORB features from an image.
        """
        orb = cv2.ORB_create(nfeatures=max_features)
        keypoints_coords = []
        if grid_size is None:
            keypoints = orb.detect(image, None)
            keypoints_coords = [(int(kp.pt[0]), int(kp.pt[1])) for kp in keypoints]
        else:
            rows, cols = grid_size
            h, w = image.shape[:2]
            grid_h, grid_w = h // rows, w // cols
            for i in range(rows):
                for j in range(cols):
                    x_start, x_end = j * grid_w, (j + 1) * grid_w
                    y_start, y_end = i * grid_h, (i + 1) * grid_h
                    grid_img = image[y_start:y_end, x_start:x_end]
                    keypoints = sorted(
                        orb.detect(grid_img, None),
                        key=lambda kp: kp.response,
                        reverse=True,
                    )[:max_features]
                    keypoints_coords.extend(
                        [
                            (int(kp.pt[0]) + x_start, int(kp.pt[1]) + y_start)
                            for kp in keypoints
                        ]
                    )
        return keypoints_coords

    @staticmethod
    def extract_good_features_to_track(
        image: np.ndarray,
        max_corners: int = 500,
        quality_level: float = 0.01,
        min_distance: int = 10,
    ) -> List[Tuple[int, int]]:
        """
        Extract good features to track using OpenCV.
        """
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        corners = cv2.goodFeaturesToTrack(
            gray_image,
            maxCorners=max_corners,
            qualityLevel=quality_level,
            minDistance=min_distance,
        )
        return (
            [(int(c[0][0]), int(c[0][1])) for c in corners]
            if corners is not None
            else []
        )

    @staticmethod
    def calc_tracks_stats(
        pred_visibility: torch.Tensor,
    ) -> Tuple[int, float, torch.Tensor]:
        """
        Calculate track statistics.
        """
        num_frames, num_tracks = pred_visibility.shape
        visible_tracks_last_frame = pred_visibility[-1]
        num_visible_tracks = int(np.sum(visible_tracks_last_frame))
        percentage_visible = (num_visible_tracks / num_tracks) * 100
        reliability = np.sum(pred_visibility, axis=0) / num_frames
        return num_visible_tracks, percentage_visible, reliability

    def filter_outlier_tracks(
        self,
        pred_3d_tracker: np.ndarray,
        pred_visibility: np.ndarray,
        eps: float = 0.5,
        min_samples: int = 5,
        frame_idx: int = None,
        use_all_frames: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Filter out tracks that are far away from the majority of tracks using DBSCAN clustering.

        Args:
            pred_3d_tracker: Array of shape [num_frames, num_tracks, 3] containing 3D track coordinates
            pred_visibility: Boolean array of shape [num_frames, num_tracks] indicating track visibility
            eps: DBSCAN parameter for maximum distance between neighboring points
            min_samples: DBSCAN parameter for minimum number of points to form a core cluster
            frame_idx: Specific frame to use for clustering (if None, a frame with most visible points is used)
            use_all_frames: Whether to use points from all frames for clustering instead of a single frame

        Returns:
            Tuple containing filtered 3D tracker array and visibility array
        """

        num_frames, num_tracks, _ = pred_3d_tracker.shape

        # If no specific frame provided and not using all frames, find frame with most visible points
        if frame_idx is None and not use_all_frames:
            visible_counts = np.sum(pred_visibility, axis=1)
            frame_idx = np.argmax(visible_counts)
            loguru.logger.info(
                f"Using frame {frame_idx} with {visible_counts[frame_idx]} visible points for clustering"
            )

        # Collect points for clustering
        if use_all_frames:
            # Use points from all frames
            points_for_clustering = []
            track_indices = []

            for f in range(num_frames):
                for t in range(num_tracks):
                    if pred_visibility[f, t]:
                        points_for_clustering.append(pred_3d_tracker[f, t])
                        track_indices.append(t)

            if not points_for_clustering:
                loguru.logger.warning("No visible points found across all frames")
                return pred_3d_tracker, pred_visibility

            points_for_clustering = np.array(points_for_clustering)
            track_indices = np.array(track_indices)
        else:
            # Use points from a single frame
            visible_tracks = pred_visibility[frame_idx]
            if not np.any(visible_tracks):
                loguru.logger.warning(f"No visible points found in frame {frame_idx}")
                return pred_3d_tracker, pred_visibility

            points_for_clustering = pred_3d_tracker[frame_idx, visible_tracks]
            track_indices = np.where(visible_tracks)[0]

        # Apply DBSCAN clustering
        clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(points_for_clustering)

        # Find the largest cluster (ignoring noise labeled as -1)
        labels = clustering.labels_
        unique_labels, counts = np.unique(labels[labels != -1], return_counts=True)

        if len(unique_labels) == 0:
            loguru.logger.warning("No clusters found, try adjusting DBSCAN parameters")
            return pred_3d_tracker, pred_visibility

        main_cluster = unique_labels[np.argmax(counts)]
        loguru.logger.info(
            f"Found {len(unique_labels)} clusters. Using main cluster with {np.max(counts)} points"
        )

        # Get indices of points in the main cluster
        if use_all_frames:
            # Create a set of track indices in main cluster
            main_cluster_tracks = set()
            for i, label in enumerate(labels):
                if label == main_cluster:
                    main_cluster_tracks.add(track_indices[i])

            # Convert to mask for all tracks
            main_cluster_mask = np.zeros(num_tracks, dtype=bool)
            for track_idx in main_cluster_tracks:
                main_cluster_mask[track_idx] = True
        else:
            # Convert to mask for all tracks
            main_cluster_mask = np.zeros(num_tracks, dtype=bool)
            main_cluster_mask[track_indices[labels == main_cluster]] = True

        # Filter 3D tracks and visibility
        pred_3d_tracker_filtered = pred_3d_tracker[:, main_cluster_mask]
        pred_visibility_filtered = pred_visibility[:, main_cluster_mask]

        loguru.logger.info(
            f"Kept {np.sum(main_cluster_mask)} tracks in the main cluster, "
            f"filtered out {np.sum(~main_cluster_mask)} outlier tracks"
        )

        return pred_3d_tracker_filtered, pred_visibility_filtered

    @staticmethod
    def filter_occluded_tracks(
        pred_3d_tracker: np.ndarray,
        pred_visibility: np.ndarray,
        occlusion_threshold: float = 0.5,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Filter out tracks that are occluded for more than a specified percentage of frames.

        Args:
            pred_3d_tracker: Array of shape [num_frames, num_tracks, 3] containing 3D track coordinates
            pred_visibility: Boolean array of shape [num_frames, num_tracks] indicating track visibility
            occlusion_threshold: Float between 0 and 1, tracks with occlusion percentage higher than
                                 this threshold will be removed

        Returns:
            Tuple containing filtered 3D tracker array and visibility array
        """
        num_frames = pred_visibility.shape[0]
        visibility_percentage = np.sum(pred_visibility, axis=0) / num_frames
        visible_enough = visibility_percentage >= (1.0 - occlusion_threshold)

        loguru.logger.info(
            f"Keeping {np.sum(visible_enough)} tracks with occlusion rate below {occlusion_threshold*100:.1f}%, "
            f"removing {np.sum(~visible_enough)} tracks"
        )

        pred_3d_tracker_filtered = pred_3d_tracker[:, visible_enough]
        pred_visibility_filtered = pred_visibility[:, visible_enough]

        return pred_3d_tracker_filtered, pred_visibility_filtered


def build_difference_matrix(size, order):
    """
    Builds a sparse difference matrix of a given order for a single sequence.

    Args:
        size (int): The length of the sequence (T).
        order (int): The order of the difference (1 for velocity, 2 for acceleration, 3 for jerk).

    Returns:
        scipy.sparse.csr_matrix: The difference matrix.
                                 Shape depends on order (e.g., (size-order) x size).
    """
    if order == 1:  # Velocity: p[t] - p[t-1]
        diagonals = [-np.ones(size), np.ones(size)]
        offsets = [0, 1]
        # Matrix maps p' to velocities. Shape (T-1) x T
        D = sp.diags(diagonals, offsets, shape=(size - 1, size), format="csr")
        # We remove the last row which computes p[T]-p[T-1] and depends on p[T] which doesn't exist
        # Correct matrix should have -1 at (i,i) and 1 at (i, i+1)
        diagonals = [-np.ones(size - 1), np.ones(size - 1)]
        offsets = [0, 1]
        D = sp.diags(diagonals, offsets, shape=(size - 1, size), format="csr")

    elif order == 2:  # Acceleration: p[t+1] - 2p[t] + p[t-1]
        diagonals = [np.ones(size), -2 * np.ones(size), np.ones(size)]
        offsets = [0, 1, 2]
        # Matrix maps p' to accelerations. Shape (T-2) x T
        D = sp.diags(diagonals, offsets, shape=(size - 2, size), format="csr")

    elif order == 3:  # Jerk: p[t+2] - 3p[t+1] + 3p[t] - p[t-1]
        # Ensure sufficient length
        if size < 4:
            # Cannot compute 3rd order difference for sequence < 4
            return sp.csr_matrix((0, size))
        diagonals = [
            -np.ones(size),
            3 * np.ones(size),
            -3 * np.ones(size),
            np.ones(size),
        ]
        offsets = [0, 1, 2, 3]
        # Matrix maps p' to jerks. Shape (T-3) x T
        D = sp.diags(diagonals, offsets, shape=(size - 3, size), format="csr")

    else:
        raise ValueError("Order must be 1, 2, or 3")

    return D


def smooth_trajectory_optimization(points, visibility, lambda_vel=0.1, lambda_jerk=1.0):
    """
    Smooths 3D point trajectories using optimization, minimizing a cost function
    combining data fidelity, velocity penalty, and jerk penalty.

    Args:
        points (np.ndarray): A NumPy array of shape (num_frames, num_points, 3)
                             containing the observed 3D coordinates.
        visibility (np.ndarray): A NumPy array of shape (num_frames, num_points)
                                 containing boolean or binary (1/0) visibility flags.
                                 True or 1 means the point is visible.
        lambda_vel (float): Weight for the velocity regularization term (1st order difference).
                            Controls smoothing based on velocity changes.
        lambda_jerk (float): Weight for the jerk regularization term (3rd order difference).
                             Controls smoothing based on acceleration changes.

    Returns:
        np.ndarray: A NumPy array of shape (num_frames, num_points, 3)
                    containing the smoothed 3D coordinates.
                    Returns None if inputs are invalid (e.g., insufficient frames for jerk).
    """
    if points.ndim != 3 or points.shape[2] != 3:
        raise ValueError(
            "`points` must be a 3D array with shape (num_frames, num_points, 3)"
        )
    if visibility.ndim != 2 or visibility.shape != points.shape[:2]:
        raise ValueError(
            "`visibility` must be a 2D array with shape (num_frames, num_points)"
        )
    if points.shape[0] < 1 or points.shape[1] < 1:
        print("Warning: Empty points array provided.")
        return points.copy()  # Return empty/original if no data

    num_frames, num_points, _ = points.shape

    # Jerk requires at least 4 frames
    if lambda_jerk > 0 and num_frames < 4:
        print(
            f"Warning: num_frames ({num_frames}) < 4. Cannot compute jerk penalty. "
            f"Setting lambda_jerk to 0 for this run."
        )
        lambda_jerk = 0.0
    # Velocity requires at least 2 frames
    if lambda_vel > 0 and num_frames < 2:
        print(
            f"Warning: num_frames ({num_frames}) < 2. Cannot compute velocity penalty. "
            f"Setting lambda_vel to 0 for this run."
        )
        lambda_vel = 0.0

    # Flatten the data: Reshape points and visibility so that all frames for point 0
    # come first, then all frames for point 1, etc. (Point-major order)
    # New shape: (N * T, 3) for points, (N * T,) for visibility
    points_flat = np.transpose(points, (1, 0, 2)).reshape(-1, 3)
    visibility_flat = (
        np.transpose(visibility).flatten().astype(float)
    )  # Ensure float for V matrix

    N = num_points
    T = num_frames
    NT = N * T  # Total number of variables per coordinate (X, Y, or Z)

    # 1. Build the Data Fidelity Matrix (V)
    # Diagonal matrix with visibility flags (1 where visible, 0 where not)
    V = sp.diags(visibility_flat, format="csc")

    # 2. Build the Smoothness Penalty Matrices (L_vel, L_jerk)
    I_N = sp.identity(N, format="csc")  # Identity matrix for points

    L_vel = sp.csc_matrix((NT, NT))  # Initialize as empty sparse matrix
    if lambda_vel > 0 and T >= 2:
        D1_single = build_difference_matrix(T, 1)  # Shape (T-1, T)
        # L1 = D1^T * D1 gives the squared velocity penalty matrix for a single trajectory
        L1_single = D1_single.T @ D1_single  # Shape (T, T)
        L_vel = sp.kron(I_N, L1_single, format="csc")  # Apply to all points

    L_jerk = sp.csc_matrix((NT, NT))  # Initialize as empty sparse matrix
    if lambda_jerk > 0 and T >= 4:
        D3_single = build_difference_matrix(T, 3)  # Shape (T-3, T)
        # L3 = D3^T * D3 gives the squared jerk penalty matrix for a single trajectory
        L3_single = D3_single.T @ D3_single  # Shape (T, T)
        L_jerk = sp.kron(I_N, L3_single, format="csc")  # Apply to all points

    # 3. Construct the overall system matrix A
    # A = V + lambda_vel * L_vel + lambda_jerk * L_jerk
    A = V + lambda_vel * L_vel + lambda_jerk * L_jerk
    # Ensure A is in CSC format for efficient solving with spsolve
    A = A.tocsc()

    # Add small regularization to diagonal for numerical stability if needed,
    # especially if a point is never visible AND lambdas are zero.
    # A += 1e-9 * sp.identity(NT, format='csc') # Optional small regularization

    # 4. Solve the linear system A * p'_smooth = V * p_orig for each dimension (X, Y, Z)
    smoothed_points_flat = np.zeros_like(points_flat)

    for dim in range(3):
        p_orig_flat = points_flat[:, dim]
        # Construct the right-hand side: b = V * p_orig
        # Only considers visible points for the data term target
        b = V @ p_orig_flat

        # Solve the sparse linear system
        try:
            p_smooth_flat = spsolve(A, b)
            smoothed_points_flat[:, dim] = p_smooth_flat
        except Exception as e:
            print(f"Error solving linear system for dimension {dim}: {e}")
            print("Returning original points for this dimension.")
            smoothed_points_flat[:, dim] = p_orig_flat

    # 5. Reshape the smoothed points back to the original format (T, N, 3)
    smoothed_points = smoothed_points_flat.reshape(N, T, 3)
    smoothed_points = np.transpose(smoothed_points, (1, 0, 2))  # Back to (T, N, 3)

    return smoothed_points
