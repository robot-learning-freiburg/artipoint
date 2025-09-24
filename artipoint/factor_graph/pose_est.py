import numpy as np
import os
import sys
import gtsam
import typing
from functools import partial
from gtsam import symbol_shorthand
from scipy.spatial.transform import Rotation

from artipoint.utils.articulation_helper import (
    estimate_motion_axis_eigen,
    estimate_motion_point_bisectors,
    calculate_error,
    angle_between_vectors,
    point_line_dist,
    plot_results,
    plot_results_o3d,
)

# Symbol shortcuts for better readability
X = symbol_shorthand.X  # articulation variables
T = symbol_shorthand.T  # SE(3) relative transformations from p to q at time i
P = symbol_shorthand.P  # Point variables
Q = symbol_shorthand.Q  # Point variables
th = symbol_shorthand.H  # Articulation variables
T_w = symbol_shorthand.W  # World frame transformation


class PoseEstFactorGraph:
    def __init__(
        self,
        num_frames: int,
        pose_noise_sigma: float = 0.1,
        point_noise_sigma: float = 0.1,
        xi_noise_sigma: float = 0.1,
        theta_noise_sigma: float = 25,
    ):
        """
        Initialize the factor graph for SE(3) transformations

        Args:
            num_frames: Number of consecutive frames
            pose_noise_sigma: Noise standard deviation for SE(3) transformations
            point_noise_sigma: Noise standard deviation for 3D points
            xi_noise_sigma: Noise standard deviation for the articulation twist vector
            theta_noise_sigma: Noise standard deviation for the articulation angle in degrees
        """
        self.full_factor_graph_ = gtsam.NonlinearFactorGraph()
        self.full_initial_values_ = gtsam.Values()
        self.num_frames = num_frames
        self.point_pairs = []

        # Configure optimization parameters
        self.fg_params_ = gtsam.LevenbergMarquardtParams()
        # self.fg_params_.setVerbosityLM("SUMMARY")
        # Uncomment and adjust these parameters if needed
        # self.fg_params_.setMaxIterations(200)
        # self.fg_params_.setlambdaInitial(1)
        # self.fg_params_.setRelativeErrorTol(1e-8)
        # self.fg_params_.setAbsoluteErrorTol(1e-8)

        # Articulation noise models
        theta_sigma = np.array([np.deg2rad(theta_noise_sigma)])
        self.theta_noise_model = gtsam.noiseModel.Diagonal.Sigmas(theta_sigma)
        self.xi_noise_model = gtsam.noiseModel.Diagonal.Sigmas(
            np.array([xi_noise_sigma] * 6)
        )

        # Noise models for SE(3) transformations
        pose_sigmas = np.array([pose_noise_sigma] * 6)
        self.pose_noise_model_6D = gtsam.noiseModel.Diagonal.Sigmas(pose_sigmas)

        # Robust noise model with Huber loss for SE(3) transformations
        self.huber_pose_noise_6D = gtsam.noiseModel.Robust.Create(
            gtsam.noiseModel.mEstimator.Huber.Create(1.345), self.pose_noise_model_6D
        )

        # Noise models for 3D points
        point_sigmas = np.array([point_noise_sigma] * 3)
        self.point_noise_model_3D = gtsam.noiseModel.Diagonal.Sigmas(point_sigmas)

        # Robust noise model with Huber loss for 3D points
        self.huber_point_noise_3D = gtsam.noiseModel.Robust.Create(
            gtsam.noiseModel.mEstimator.Huber.Create(1.345), self.point_noise_model_3D
        )

        # Pre-allocate Jacobian matrices for repeated operations
        self._j_T_w_0 = np.zeros((6, 6), order="F")
        self._j_T_0_i = np.zeros((6, 6), order="F")
        self._j_P_i = np.zeros((3, 3), order="F")
        self._j_T_w_i = np.zeros((3, 6), order="F")
        self._j_xi_th = np.zeros((6, 6), order="F")
        self._j_T_twist = np.zeros((6, 6), order="F")
        self._j_T_i = np.zeros((6, 6), order="F")
        self._j_error = np.zeros((6, 6), order="F")
        self._j_T_i_point = np.zeros((3, 6), order="F")
        self._j_p = np.zeros((3, 3), order="F")

        # Cache for computed values
        self._optimizer_result_ = None

    def reset(self):
        """Reset the estimator state"""
        self.full_factor_graph_ = gtsam.NonlinearFactorGraph()
        self.full_initial_values_ = gtsam.Values()
        self.point_pairs = []
        self._optimizer_result_ = None

    def add_point_pair_factor(self, point_pairs: np.ndarray, frame_idx: int):
        """
        Add point pair factors between consecutive frames

        Args:
            point_pairs: Numpy array of 3D point pairs with shape (n, 2, 3)
                        where n is number of points, first dim is p/q, second is xyz
            frame_idx: Index of the current frame
        """
        if frame_idx >= self.num_frames:
            return

        # Extract p and q arrays
        p_array = point_pairs[:, 0, :]  # shape: (n, 3)
        q_array = point_pairs[:, 1, :]  # shape: (n, 3)

        # create huper loss noise model based on dimension of the point pairs
        point_noise_sigma = 0.1
        point_sigmas = np.array([point_noise_sigma] * 3 * p_array.shape[0])
        huber_point_noise = gtsam.noiseModel.Robust.Create(
            gtsam.noiseModel.mEstimator.Huber.Create(1.345),
            gtsam.noiseModel.Diagonal.Sigmas(point_sigmas),
        )

        # Create a single factor for all point pairs
        point_pair_factor = gtsam.CustomFactor(
            huber_point_noise,
            [T(frame_idx)],
            partial(self._point_pair_error, p_array, q_array),
        )
        self.full_factor_graph_.add(point_pair_factor)

        # Insert initial transformations (identity if unknown)
        if not self.full_initial_values_.exists(T(frame_idx)):
            self.full_initial_values_.insert(T(frame_idx), gtsam.Pose3())

        # Add point pairs to the list
        self.point_pairs.append(point_pairs)

    def add_articulation_factor(
        self, frame_idx: int, theta_prior: float = 1e-6, single_xi=True
    ):
        """
        Add an articulation factor between consecutive frames

        Args:
            frame_idx: Index of the current frame
            theta_prior: Prior for the articulation angle in degrees
        """
        if frame_idx >= self.num_frames:
            return

        # Add prior for the first articulation 6D vector (only once)
        if frame_idx == 0 or not single_xi:
            self.full_initial_values_.insert(X(frame_idx), np.zeros(6))

        # Create articulation factor
        articulation_factor = gtsam.CustomFactor(
            self.huber_pose_noise_6D,
            [
                X(frame_idx * (not single_xi)),
                T(frame_idx),
                th(frame_idx),
            ],  # Hacks from hell
            partial(self._articulation_factor),
        )
        self.full_factor_graph_.add(articulation_factor)

        # Insert prior for the theta
        if theta_prior is not None:
            self.full_factor_graph_.add(
                gtsam.PriorFactorDouble(
                    th(frame_idx), np.deg2rad(theta_prior), self.theta_noise_model
                )
            )
            # Insert initial articulation angle
            self.full_initial_values_.insert(th(frame_idx), theta_prior)
        else:
            self.full_initial_values_.insert(th(frame_idx), 1e-6)

    def add_world_frame_factor(self, frame_idx: int):
        """
        Add a world frame factor for the first frame
        Estimate the SE3 pose of the first observation of the object (group of points) in the camera frame

        Args:
            frame_idx: Index of the current frame
        """
        if frame_idx >= self.num_frames:
            return

        if frame_idx == 0:
            # Add prior for the first world frame transformation
            self.full_factor_graph_.add(
                gtsam.PriorFactorPose3(T_w(0), gtsam.Pose3(), self.huber_pose_noise_6D)
            )
            # Insert initial world frame transformation
            self.full_initial_values_.insert(T_w(0), gtsam.Pose3())

            # Create world frame factor
            for i in range(self.point_pairs[0].shape[0]):
                q = self.point_pairs[0][i][0]
                world_frame_factor = gtsam.CustomFactor(
                    self.huber_point_noise_3D,
                    [T_w(0)],
                    partial(self._world_frame_factor, q),
                )
                self.full_factor_graph_.add(world_frame_factor)

    def _world_frame_factor(
        self,
        q: np.ndarray,
        this: gtsam.CustomFactor,
        values: gtsam.Values,
        jacobians: typing.Optional[typing.List[np.ndarray]] = None,
    ) -> np.ndarray:
        """
        Calculate error for a world frame factor

        Args:
            T_0_i: Transform from frame 0 to frame i
            this: Custom factor instance
            values: Current values in the factor graph
            jacobians: Optional Jacobian matrices

        Returns:
            Error vector
        """
        keys = this.keys()
        T_0_w = values.atPose3(keys[0])  # from world to frame 0

        zero_point = gtsam.Point3(np.zeros(3))
        Q_w = T_0_w.transformFrom(zero_point, self._j_T_w_i, self._j_P_i)

        if jacobians is not None:
            jacobians[0] = self._j_T_w_i

        # Use the first point pair in the latest frame for error calculation
        if not self.point_pairs:
            return np.zeros(3)

        try:
            residual = Q_w - q
            return residual
        except IndexError:
            print("Warning: No point pairs available for world frame factor")
            return np.zeros(3)

    def _articulation_factor(
        self,
        this: gtsam.CustomFactor,
        values: gtsam.Values,
        jacobians: typing.Optional[typing.List[np.ndarray]] = None,
    ) -> np.ndarray:
        """
        Calculate error for an articulation factor

        error = T_w(0) . Exp(th(i)*X(i)) relative to T(i)

        Args:
            this: Custom factor instance
            values: Current values in the factor graph
            jacobians: Optional Jacobian matrices

        Returns:
            Error vector
        """
        keys = this.keys()
        X_i = values.atVector(keys[0])
        T_i = values.atPose3(keys[1])
        th_i = values.atDouble(keys[2])

        # Use pre-allocated Jacobians
        T_twist = gtsam.Pose3.Expmap(X_i * th_i, self._j_xi_th)
        T_diff = T_twist.between(T_i, self._j_T_twist, self._j_T_i)
        error = gtsam.Pose3.Logmap(T_diff, self._j_error)

        if jacobians is not None:
            # Apply chain rule to compute Jacobians
            jacobians[0] = (
                self._j_error @ self._j_T_twist @ self._j_xi_th @ (th_i * np.eye(6))
            )  # wrt X
            jacobians[1] = self._j_error @ self._j_T_i  # wrt T_i
            jacobians[2] = (
                self._j_error @ self._j_T_twist @ self._j_xi_th @ X_i
            )  # wrt theta

        return error

    def _point_pair_error(
        self,
        p_array: np.ndarray,
        q_array: np.ndarray,
        this: gtsam.CustomFactor,
        values: gtsam.Values,
        jacobians: typing.Optional[typing.List[np.ndarray]] = None,
    ) -> np.ndarray:
        """
        Calculate error for a batch of point pairs

        Args:
            p_array: Array of 3D points in frame i, shape (n, 3)
            q_array: Array of 3D points in frame i+1, shape (n, 3)
            this: Custom factor instance
            values: Current values in the factor graph
            jacobians: Optional Jacobian matrices

        Returns:
            Error vector, shape (n*3,)
        """
        keys = this.keys()
        T_i = values.atPose3(keys[0])

        # Extract rotation and translation from pose
        R = T_i.rotation().matrix()
        t = T_i.translation()

        # Transform all points
        p_transformed = np.dot(p_array, R.T) + t

        # Calculate residual
        residual = p_transformed - q_array

        if jacobians is not None:
            # Manually calculate Jacobian
            num_points = p_array.shape[0]
            jacobian = np.zeros((num_points * 3, 6))

            for i in range(num_points):
                p = p_array[i]

                # Jacobian block for this point (3x6)
                point_jac = np.zeros((3, 6))

                # Rotation part (3x3): -R * skew(p)
                skew_p = np.array(
                    [[0, -p[2], p[1]], [p[2], 0, -p[0]], [-p[1], p[0], 0]]
                )
                point_jac[:, :3] = -np.dot(R, skew_p)

                # Translation part (3x3): Identity
                point_jac[:, 3:] = np.eye(3)

                # Place in full Jacobian
                jacobian[i * 3 : (i + 1) * 3, :] = point_jac

            jacobians[0] = jacobian

        # Flatten residual to match expected shape
        return residual.flatten()

    def solve_factor_graph(self) -> bool:
        """
        Solve the factor graph using Levenberg-Marquardt optimization

        Returns:
            bool: True if optimization succeeded, False otherwise
        """
        if len(self.point_pairs) == 0:
            print("Error: No point pairs added to factor graph")
            return False

        optimizer = gtsam.LevenbergMarquardtOptimizer(
            self.full_factor_graph_, self.full_initial_values_, self.fg_params_
        )

        try:
            self._optimizer_result_ = optimizer.optimize()
            return True
        except Exception as exc:
            print(f"Optimization failed: {exc}")
            return False

    def get_optimized_transformations(self):
        """
        Retrieve optimized transformations

        Returns:
            List of optimized GTSAM Pose3 transformations
        """
        if self._optimizer_result_ is None:
            print("Warning: Factor graph not solved yet")
            return []

        return [
            self._optimizer_result_.atPose3(T(i)) for i in range(len(self.point_pairs))
        ]

    def get_optimized_points(self):
        """
        Retrieve optimized points

        Returns:
            List of optimized 3D points
        """
        if self._optimizer_result_ is None:
            print("Warning: Factor graph not solved yet")
            return []

        result = []
        for i in range(len(self.point_pairs)):
            try:
                result.append(
                    [
                        self._optimizer_result_.atPoint3(P(i)),
                        self._optimizer_result_.atPoint3(Q(i)),
                    ]
                )
            except RuntimeError:
                # Skip points that don't exist in the optimization result
                pass
        return np.asarray(result) if result else np.array([])

    def get_optimized_thetas(self):
        """
        Retrieve optimized articulation angles

        Returns:
            List of optimized theta values
        """
        if self._optimizer_result_ is None:
            print("Warning: Factor graph not solved yet")
            return []

        return [
            self._optimizer_result_.atDouble(th(i))
            for i in range(len(self.point_pairs))
        ]

    def get_optimized_twist(self):
        """
        Retrieve optimized articulation twist vector

        Returns:
            6D vector representing the optimized articulation twist
        """
        if self._optimizer_result_ is None:
            print("Warning: Factor graph not solved yet")
            return np.zeros(6)

        return self._optimizer_result_.atVector(X(0))

    def get_optimized_world_frame(self):
        """
        Retrieve optimized world frame transformation

        Returns:
            Optimized GTSAM Pose3 world frame transformation
        """
        if self._optimizer_result_ is None:
            print("Warning: Factor graph not solved yet")
            return np.eye(4)

        try:
            return self._optimizer_result_.atPose3(T_w(0)).matrix()
        except RuntimeError:
            print("Warning: World frame transformation not in optimization result")
            return np.eye(4)

    def estimate_joint_type_and_parameters(self):
        """
        Analyze the optimized results to determine joint type and parameters

        Returns:
            dict: Joint type and parameters
        """
        if self._optimizer_result_ is None:
            print("Warning: Factor graph not solved yet")
            return {"type": "unknown"}

        result = {}
        twist = self.get_optimized_twist()
        w_norm = np.linalg.norm(twist[:3])
        v_norm = np.linalg.norm(twist[3:])

        thetas = self.get_optimized_thetas()
        traj_twist = [self.get_optimized_world_frame()]
        for i in range(len(self.point_pairs)):
            Tw = gtsam.Pose3.Expmap(twist * thetas[i]).matrix()
            traj_twist.append(np.linalg.inv(Tw) @ traj_twist[-1])
        result["traj_twist"] = traj_twist
        result["thetas"] = thetas
        result["twist"] = twist

        # calculate the pitch
        w, v = twist[:3], twist[3:]
        pitch = np.dot(w, v) / np.linalg.norm(w) ** 2
        result["pitch"] = pitch

        if np.abs(pitch) > 0.2:
            result["type"] = "prismatic"
            axis = twist[3:] / v_norm if v_norm > 1e-6 else np.array([0, 1, 0])
            result["axis"] = axis
            result["center"] = self.get_optimized_world_frame()[:3, 3]
        else:
            axis = twist[:3] / w_norm
            center = estimate_motion_point_bisectors(traj_twist, axis)
            result["type"] = "revolute"
            result["axis"] = axis
            result["center"] = center
            q = np.cross(w, v) / np.linalg.norm(w) ** 2
            result["q"] = q

        return result

    def export_6D_poses(self, filename: str):
        """
        Export the optimized 6D poses to a file

        Args:
            filename: Path to the output file
        """
        if self._optimizer_result_ is None:
            print("Warning: Factor graph not solved yet")
            return
        T_w = self.get_optimized_world_frame()
        free_poses = [T_w]
        for i in range(self.num_frames):
            pose = self._optimizer_result_.atPose3(T(i)).matrix()
            w_T_i = np.linalg.inv(pose) @ free_poses[-1]
            free_poses.append(w_T_i)

        with open(filename, "w") as f:
            # write the csv header
            f.write("frame, x, y, z, qx, qy, qz, qw\n")
            for i, pose in enumerate(free_poses):
                x, y, z = pose[:3, 3]
                q = Rotation.from_matrix(pose[:3, :3]).as_quat()
                f.write(f"{i}, {x}, {y}, {z}, {q[0]}, {q[1]}, {q[2]}, {q[3]}\n")
        print(f"Exported 6D poses to {filename}")
        return True

    def load_6D_poses(self, filename: str):
        """
        Load 6D poses from a file

        Args:
            filename: Path to the input file
        """
        if not os.path.exists(filename):
            print(f"File {filename} does not exist")
            return False

        poses = []
        with open(filename, "r") as f:
            lines = f.readlines()[1:]
            for line in lines:
                parts = line.strip().split(",")
                frame = int(parts[0])
                x, y, z = map(float, parts[1:4])
                qx, qy, qz, qw = map(float, parts[4:])
                pose = np.eye(4)
                pose[:3, 3] = [x, y, z]
                pose[:3, :3] = Rotation.from_quat([qx, qy, qz, qw]).as_matrix()
                poses.append(pose)
        return np.array(poses)
