import numpy as np
import torch
import cv2
import loguru
from tqdm import tqdm
from pathlib import Path
import sys
import os

from hands_segmentation.hand_segmentor import HandSegmentor
from utils.mobile_sam_segmentor import MobileSAMSegmenter


class ArticulatedObjectSegmentor:
    """Segmentation of articulated objects using hand detection and SAM."""

    def __init__(self, config):
        """
        Initialize the ArticulatedObjectSegmentor.

        Args:
            config (dict): Configuration dictionary containing model parameters.
        """
        self.device = torch.device(
            "cuda"
            if config.get("use_cuda", True) and torch.cuda.is_available()
            else "cpu"
        )

        # Initialize Hand Segmentor
        self.hand_segmentor = HandSegmentor(
            model_repo=config["hand_model_repo"],
            model_name=config["hand_model_name"],
            model_checkpoint_path=config["hand_model_checkpoint_path"],
            resize=config.get("hand_resize", (256, 256)),
            use_cuda=config.get("use_cuda", True),
        )

        # Initialize MobileSAMSegmenter
        sam_config = {
            "checkpoint": config["sam_checkpoint"],
            "model_type": config.get("sam_model_type", "vit_t"),
            "device": self.device,
        }
        self.sam_segmenter = MobileSAMSegmenter(config=sam_config)

        # Arti4D dataset for 3D processing
        self.arti4d_dataset = config.get("arti4d_dataset", None)

    def segment_articulated_object(
        self,
        rgb_image,
        depth_image,
        camera_pose=None,
        hand_mask_size=250,
        num_points=25,
        eps_pixels=5,
        dist_thresh=0.15,
        max_feat_points=10,
        feat_type="shi",
        sample_fetures=True,
    ):
        """
        Segment articulated objects in an image based on hand detection.

        Args:
            rgb_image (np.ndarray): RGB image input.
            depth_image (np.ndarray, optional): Depth image for 3D projection.
            hand_mask_size (int): Minimum size for valid hand mask.
            num_points (int): Number of points to sample near hand.
            eps_pixels (int): Epsilon for point sampling.
            dist_thresh (float): Distance threshold for filtering objects from hand.
            orb_num_points (int): Number of ORB features to sample.
            feat_type (str): Feature type for sampling (orb or good_features).
            sample_fetures (bool): Whether to sample features on filtered masks.
            max_feat_points (int): Maximum number of feature points to sample.

        Returns:
            dict: Dictionary containing segmentation results.
        """
        result = {}

        # 1. Segment hand (90 FPS with size 256x256)
        hand_mask = self.hand_segmentor.segment(rgb_image)
        hand_mask = self.hand_segmentor.remove_small_regions(
            hand_mask, 1000.0, mode="islands"
        )
        if np.sum(hand_mask) < hand_mask_size:
            return result

        # 2. Sample points near hand
        hand_indices = np.argwhere(hand_mask > 0.99)
        points = self.hand_segmentor.sample_points_near_hand(
            hand_indices, num_points, eps_pixels
        )

        # Scale points to original image size
        scale_x = rgb_image.shape[1] / hand_mask.shape[1]
        scale_y = rgb_image.shape[0] / hand_mask.shape[0]
        points = [(int(p[1] * scale_x), int(p[0] * scale_y)) for p in points]
        point_labels = [1] * len(points)  # All points are foreground

        # 3. Use points to prompt MobileSAM to segment objects
        obj_masks, scores, _ = self.sam_segmenter.segment_multiple_objects_with_points(
            rgb_image, points, point_labels
        )

        # 4. Filter and process 2D masks
        obj_masks = self._process_masks(obj_masks, scores, hand_mask)

        # Store basic segmentation results
        result["obj_masks"] = obj_masks
        result["scores"] = scores
        result["points"] = points
        result["point_labels"] = point_labels
        result["hand_mask"] = hand_mask

        # 5. Process and filter in 3D if depth is available
        if obj_masks:
            obj_pcds, filtered_2D_masks = self._filter_objects_3d(
                rgb_image, depth_image, obj_masks, hand_mask, dist_thresh, camera_pose
            )
            if obj_pcds:
                result["obj_pcds"] = obj_pcds
                result["filtered_masks"] = filtered_2D_masks

            # 6. Sample features on filtered masks
            if filtered_2D_masks and sample_fetures:
                orb_points = self.sample_fetures(
                    rgb_image, filtered_2D_masks, max_feat_points, feat_type=feat_type
                )
                result["orb_points"] = orb_points
        else:
            loguru.logger.warning("No objects detected.")
        return result

    def _process_masks(self, masks, scores, hand_mask):
        """Process and filter segmentation masks."""
        masks = self.sam_segmenter.filter_masks_by_score(masks, scores)
        masks = self.sam_segmenter.remove_small_regions(masks, 1000, mode="islands")
        masks = self.sam_segmenter.filter_masks_by_size(masks, 2000, 100000)
        masks = self.sam_segmenter.filter_redeundant_masks(masks, iou_thresh=0.25)
        masks = self.remove_hand_mask(masks, hand_mask, iou_thresh=0.01)
        return masks

    def _filter_objects_3d(
        self,
        rgb_image,
        depth_image,
        obj_masks,
        hand_mask,
        dist_thresh=0.15,
        camera_pose=None,
    ):
        """
        Process and filter objects based on their 3D distance from the hand.

        Args:
            rgb_image (np.ndarray): RGB image.
            depth_image (np.ndarray): Depth image.
            obj_masks (list): Object masks.
            hand_mask (np.ndarray): Hand mask.
            dist_thresh (float): Distance threshold.

        Returns:
            tuple: (obj_pcds, filtered_2D_masks)
        """
        # Project masks to 3D
        obj_pcds = self.project_mask_3d(rgb_image, depth_image, obj_masks, camera_pose)
        hand_pcd = self.project_mask_3d(
            rgb_image, depth_image, [hand_mask], camera_pose
        )[0]

        # Filter objects by distance from hand
        obj_pcds, idx = self.remove_objects_far_from_hand(
            hand_pcd, obj_pcds, dist_thrsh=dist_thresh
        )

        if not obj_pcds:
            return None, None

        # Get corresponding masks
        filtered_2D_masks = [obj_masks[i] for i in idx]

        return obj_pcds, filtered_2D_masks

    def sample_fetures(self, rgb_image, masks, num_points=10, feat_type="shi"):
        """
        Sample features on the segmented masks.

        Args:
            rgb_image (np.ndarray): RGB image.
            masks (list): List of binary masks.
            num_points (int): Number of points to sample.
            feat_type (str): Feature type (orb or good_features).

        Returns:
            list: List of sampled feature points.
        """

        track_points = []
        for obj_mask in masks:
            if feat_type == "orb":
                points = self.sample_orb_features(
                    rgb_image, obj_mask.astype(np.uint8) * 255, nfeatures=num_points
                )
            else:
                points = self.sample_good_features_to_track(
                    rgb_image, obj_mask.astype(np.uint8) * 255, max_corners=num_points
                )
            track_points.append(points)

        # Flatten point lists
        track_points = [p for sublist in track_points for p in sublist]

        return track_points

    def remove_hand_mask(self, masks, hand_mask, iou_thresh=0.5):
        """
        Removes masks that overlap with the hand mask based on IoU threshold.

        Args:
            masks (list): List of binary masks to filter.
            hand_mask (np.ndarray): Binary hand mask.
            iou_thresh (float): IoU threshold for filtering.

        Returns:
            list: Filtered masks.
        """
        if not masks:
            return []

        # Convert hand mask to proper format
        if hand_mask.dtype == np.bool:
            hand_mask = hand_mask.astype(np.uint8)

        # Resize hand mask to match other masks
        hand_mask_resized = cv2.resize(
            hand_mask,
            (masks[0].shape[1], masks[0].shape[0]),
            interpolation=cv2.INTER_NEAREST,
        )
        hand_mask_binary = hand_mask_resized > 0.5

        # Filter masks by IoU
        filtered_masks = []
        for mask in masks:
            iou = self.sam_segmenter.compute_iou(mask, hand_mask_binary)
            if iou <= iou_thresh:
                filtered_masks.append(mask)

        return filtered_masks

    def project_mask_3d(self, rgb, depth, masks, pose=None):
        """
        Projects 2D masks to 3D using depth image and camera parameters.

        Args:
            rgb (np.ndarray): RGB image.
            depth (np.ndarray): Depth image.
            masks (list): List of binary masks.
            poses (np.ndarray, optional): Camera poses.

        Returns:
            list: List of 3D point clouds.
        """
        if not masks or self.arti4d_dataset is None:
            return []

        projected_masks = []
        for i, mask in enumerate(masks):

            # Prepare mask
            mask = mask.astype(np.uint8)
            mask = cv2.resize(
                mask, (depth.shape[1], depth.shape[0]), interpolation=cv2.INTER_NEAREST
            )
            mask = mask > 0.5

            # Create point cloud
            pcd, _ = self.arti4d_dataset.create_pcd(
                rgb, depth, camera_pose=pose, maskout=~mask
            )
            projected_masks.append(pcd)

        return projected_masks

    def remove_objects_far_from_hand(self, hand_pcd, objects_pcd, dist_thrsh=0.1):
        """
        Removes 3D masks that are far from the hand

        Args:
            hand_3d_mask (o3d.geometry.PointCloud): 3D mask of the hand.
            objects_3d_masks (list of o3d.geometry.PointCloud): List of 3D masks of objects.
            inter_ratio (float, optional): Chamfer distance threshold for filtering masks.
                          Masks with Chamfer distance greater than this threshold will be removed.
                          Default is 0.1.

        Returns:
            list of np.ndarray: List of filtered 3D masks that are close to the hand
        """
        filtered_masks = []
        idx = []
        for i, mask in enumerate(objects_pcd):
            z_median_hand = np.median(np.asarray(hand_pcd.points)[:, 2])
            z_median_obj = np.median(np.asarray(mask.points)[:, 2])
            diff = np.linalg.norm(z_median_hand - z_median_obj)
            if diff < dist_thrsh:
                filtered_masks.append(mask)
                idx.append(i)

        return filtered_masks, idx

    def sample_orb_features(self, image, mask, nfeatures=500):
        """
        Detect ORB keypoints and compute descriptors within the ROI.

        Args:
            image (np.ndarray): Input RGB image.
            mask (np.ndarray): Binary mask indicating ROI.
            nfeatures (int): Maximum number of features to detect.

        Returns:
            list: List of keypoint coordinates.
        """
        # Create an ORB detector
        orb = cv2.ORB_create(nfeatures=nfeatures)

        # Resize mask to match image
        mask = cv2.resize(
            mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST
        )

        # Detect keypoints and compute descriptors
        keypoints, _ = orb.detectAndCompute(image, mask)

        # Convert keypoints to (x, y) coordinates
        return [kp.pt for kp in keypoints]

    def sample_good_features_to_track(self, image, mask, max_corners=500):
        """
        Detect good features to track within the ROI.
        """
        # Convert RGB image to grayscale
        gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        # Resize mask to match image
        mask = cv2.resize(
            mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST
        )

        # Detect good features to track (using gray_image instead of image)
        corners = cv2.goodFeaturesToTrack(
            gray_image,
            maxCorners=max_corners,
            mask=mask,
            qualityLevel=0.01,
            minDistance=25,
        )

        if corners is None:
            return []

        # Convert corners to (x, y) coordinates
        return [tuple(c[0]) for c in corners]

    def sample_random_points_on_mask(self, image, mask, num_points=100):
        """
        Sample random points within the ROI defined by the mask.

        Args:
            image (np.ndarray): Input RGB image.
            mask (np.ndarray): Binary mask indicating ROI.
            num_points (int): Number of points to sample.

        Returns:
            list: List of sampled point coordinates.
        """
        # Resize mask to image size
        mask = cv2.resize(
            mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST
        )

        # Get non-zero pixels in mask
        mask_indices = np.argwhere(mask > 0.5)

        if len(mask_indices) == 0:
            return []

        # Sample random points
        indices = np.random.choice(
            len(mask_indices),
            size=min(num_points, len(mask_indices)),
            replace=len(mask_indices) < num_points,
        )
        random_points = mask_indices[indices]

        return random_points

    def extract_hand_action_segments_smoothed(
        self,
        rgb_frames,
        min_window_size=30,
        max_window_size=60,
        smoothing_window_size=10,
        threshold_avg=0.5,
        hand_mask_size=250,
    ):
        """
        Extracts segments corresponding to hand-based actions from an egocentric RGBD video sequence,
        using a moving average to smooth out intermittent false negatives.

        Args:
            rgb_frames (list): List of RGB frames.
            segmentor (ArticulatedObjectSegmentor): Instance with the segment_articulated_object method.
            min_window_size (int): Minimum number of frames for a valid hand action segment.
            smoothing_window_size (int): Number of frames over which to compute the moving average.
            threshold_avg (float): Average detection threshold (between 0 and 1) to consider the hand present.
            hand_mask_size (int): Minimum size for valid hand mask.

        Returns:
            list of tuple: List of (start_index, end_index) tuples for detected hand action segments.
        """
        segments = []
        in_segment = False
        segment_start = None
        detection_history = []

        for idx, frame in tqdm(
            enumerate(rgb_frames),
            total=len(rgb_frames),
            desc="Extracting action segments",
        ):
            hand_mask = self.hand_segmentor.segment(frame)
            hand_mask = self.hand_segmentor.remove_small_regions(
                hand_mask, 1000.0, mode="islands"
            )

            # Determine binary detection for current frame.
            detected = 0
            if np.sum(hand_mask) > hand_mask_size:
                detected = 1

            # Append the detection result to the history.
            detection_history.append(detected)
            # Use only the last 'window_size' frames for the moving average.
            window = detection_history[-smoothing_window_size:]
            avg_detection = np.mean(window) if window else 0

            # Use the smoothed result to decide if the hand is present.
            if avg_detection >= threshold_avg:
                if not in_segment:
                    # Start a new segment; adjust the start to account for the smoothing window.
                    segment_start = max(0, idx - smoothing_window_size + 1)
                    in_segment = True
            else:
                if in_segment:
                    # End the segment if the moving average falls below threshold.
                    segments.append((segment_start, idx - 1))
                    in_segment = False

        # Close any open segment at the end of the video.
        if in_segment:
            segments.append((segment_start, len(rgb_frames) - 1))

        # Filter out segments that are too short or too long.
        loguru.logger.info(f"Detected {len(segments)} segments.")
        segments = [
            seg
            for seg in segments
            if max_window_size >= seg[1] - seg[0] >= min_window_size
        ]
        loguru.logger.info(f"Filtered to {len(segments)} segments.")

        return segments

    @staticmethod
    def play_hand_action_segments(
        video_frames, segments, window_name="Hand Action Segment", frame_delay=100
    ):
        """
        Plays the segments (windows) corresponding to hand-based actions from the video frames.

        Args:
            video_frames (list): List of dictionaries representing frames (each should contain "rgb" key).
            segments (list of tuple): List of (start_index, end_index) tuples representing segments.
            window_name (str): Name of the OpenCV window.
            frame_delay (int): Delay in milliseconds between frames.
        """
        for seg_idx, (start, end) in enumerate(segments):
            loguru.logger.info(
                f"Playing segment {seg_idx+1}/{len(segments)} (frames {start} to {end}) with length {end - start + 1}"
            )
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
            for idx in range(start, end + 1):
                # Get the RGB frame, convert it to BGR for OpenCV.
                frame_rgb = video_frames[idx]
                frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

                cv2.imshow(window_name, frame_bgr)
                key = cv2.waitKey(frame_delay) & 0xFF
                if key == ord("q"):
                    # Skip the current segment if 'q' is pressed.
                    print("Skipping current segment...")
                    break
            cv2.destroyWindow(window_name)

        cv2.destroyAllWindows()
