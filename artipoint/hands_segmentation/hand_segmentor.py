import os
import glob
import torch
import torch.hub
import tqdm
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import numpy as np
import cv2
import loguru


class HandSegmentor:
    """
    Hand segmentor that extracts hand masks from input images.
    The model is loaded from a local checkpoint repository.
    """

    def __init__(
        self,
        model_repo: str = "./hands_segmentation",
        model_name: str = "hand_segmentor",
        model_checkpoint_path: str = "hands_checkpoint.ckpt",
        resize: tuple | None = None,
        use_cuda: bool = True,
    ):
        """
        Initializes the HandSegmentor.

        Parameters:
            model_repo (str): Path to the local repository for model checkpoint.
            model_name (str): Name of the model to load.
            use_cuda (bool): Whether to use CUDA device.
        """
        self.device = torch.device(
            "cuda" if use_cuda and torch.cuda.is_available() else "cpu"
        )
        kwargs = (
            {"checkpoint_path": model_checkpoint_path} if model_checkpoint_path else {}
        )

        self.model = torch.hub.load(
            repo_or_dir=model_repo,
            model=model_name,
            pretrained=True,
            source="local",
            **kwargs,
        )
        self.model.to(self.device)
        self.model.eval()
        self.resize = resize

        # Define transformation pipeline
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize(resize) if resize is not None else lambda x: x,
                # Ensure the image has 3 channels
                lambda x: x if x.shape[0] == 3 else x.repeat(3, 1, 1),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

    def segment(self, img_input) -> np.ndarray:
        """
        Segments the hand in the provided image.

        Parameters:
            img_input: A PIL.Image.Image or a numpy.ndarray image (RGB).

        Returns:
            A numpy.ndarray mask with the same width and height as the input image.
        """
        # Convert input to PIL Image if it is a numpy array
        if not isinstance(img_input, Image.Image):
            if isinstance(img_input, np.ndarray):
                img = Image.fromarray(img_input)
            else:
                raise ValueError("Input must be a PIL.Image or a numpy.ndarray.")
        else:
            img = img_input

        # Preprocess the image
        img_tensor = self.transform(img).unsqueeze(0).to(self.device)

        # Perform inference with no_grad for efficiency
        with torch.no_grad():
            logits = self.model(img_tensor)
            # Assuming the hand class is at index 1
            mask_tensor = F.softmax(logits, dim=1).squeeze(0)[1]

        return mask_tensor.detach().cpu().numpy()

    def visualize(
        self, img_input, mask: np.ndarray, points: np.ndarray = None, alpha: float = 0.2
    ) -> None:
        """
        Visualizes the original image with the hand mask overlay using OpenCV.

        Parameters:
            img_input: A PIL.Image.Image or a numpy.ndarray image (RGB).
            mask (np.ndarray): The segmentation mask.
            alpha (float): Transparency factor for the mask overlay.
        """

        # Convert input to numpy array if needed
        if isinstance(img_input, Image.Image):
            img_np = np.array(img_input)
        elif isinstance(img_input, np.ndarray):
            img_np = img_input
        else:
            raise ValueError("Input image must be a PIL.Image or a numpy.ndarray.")
        img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

        # Ensure mask is in the correct format
        if mask.ndim == 2:
            mask = (mask * 255).astype(np.uint8)
            scalex = img_np.shape[1] / mask.shape[1]
            scaley = img_np.shape[0] / mask.shape[0]
            mask = cv2.resize(
                mask,
                (img_np.shape[1], img_np.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            )

        # Create a red overlay for the mask
        red_overlay = np.zeros_like(img_np)
        red_overlay[:, :, 2] = mask  # Set the red channel

        # Overlay the mask on the image
        overlay = cv2.addWeighted(img_np, 1 - alpha, red_overlay, alpha, 1)

        # draw points
        if points is not None:
            points = [(int(p[1] * scalex), int(p[0] * scaley)) for p in points]
            for point in points:
                cv2.circle(overlay, point, 3, (0, 255, 0), -1)

        # Display the result
        cv2.imshow("Hand segmentation overlay", overlay)

    @staticmethod
    def sample_points_near_hand(hand_mask_indices, K, eps_pixel=100):
        """
        Given an array of hand mask indices (N,2), sample K random points that are
        within eps_pixel (in Euclidean distance) of the mask boundary but not inside it.

        Parameters:
        hand_mask_indices (np.ndarray): (N,2) array of (row, col) coordinates of mask pixels.
        K (int): Number of points to sample.
        eps_pixel (float): Radius (in pixels) defining the vicinity of the mask.

        Returns:
        np.ndarray: (K,2) array of (row, col) coordinates of the sampled points.
        """
        # Ensure input is a numpy array.
        hand_mask_indices = np.asarray(hand_mask_indices)
        if hand_mask_indices.shape[0] == 0:
            loguru.logger.warning("Hand mask is empty. Cannot sample points.")
            return None

        # Compute the bounding box of the mask.
        min_row = np.min(hand_mask_indices[:, 0]) - eps_pixel
        max_row = np.max(hand_mask_indices[:, 0]) + eps_pixel
        min_col = np.min(hand_mask_indices[:, 1]) - eps_pixel
        max_col = np.max(hand_mask_indices[:, 1]) + eps_pixel

        # Create a grid of points within the expanded bounding box.
        # Calculate the number of points per side
        side_points = int(np.sqrt(K))
        if side_points**2 != K:
            loguru.logger.warning(
                "K is not a perfect square. Adjusting to the nearest perfect square."
            )
            side_points = int(np.sqrt(K)) + 1

        # Create a grid of points within the expanded bounding box
        grid_x, grid_y = np.meshgrid(
            np.linspace(min_col, max_col, side_points),
            np.linspace(min_row, max_row, side_points),
        )
        grid_points = np.vstack([grid_y.ravel(), grid_x.ravel()]).T

        # Ensure we only return K points
        # sampled_points = grid_points[:K]
        return grid_points

    def find_hand_sequence(self, imgs_path, threshold=250.0):
        """
        Searches for the hand in a sequence of frames and returns the start and stop frame numbers for each appearance.
        Offline hand detection algorithm that processes the frames sequentially.
        For better performance, consider caching the results for future use.

        Parameters:
            imgs_path (list): List of image paths in the sequence.
            threshold (float): Threshold for considering a hand present in the frame.

        Returns:
            list: List of tuples (start_frame, stop_frame) where start_frame is the frame number when the hand enters the scene,
                  and stop_frame is the frame number when the hand leaves the scene.
        """
        sequences = []
        start_frame, stop_frame = None, None
        hand_present = False

        for i, img_path in tqdm.tqdm(
            enumerate(imgs_path), total=len(imgs_path), desc="Processing frames"
        ):
            image = Image.open(img_path)
            mask = self.segment(image)
            # Check if the hand is present in the frame
            if np.sum(mask) > threshold:
                if not hand_present:
                    start_frame = i
                    hand_present = True
            else:
                if hand_present:
                    stop_frame = i
                    sequences.append((start_frame, stop_frame))
                    hand_present = False

        # If the hand is present till the last frame
        if hand_present:
            stop_frame = len(imgs_path) - 1
            sequences.append((start_frame, stop_frame))

        return sequences

    @staticmethod
    def remove_small_regions(
        mask: np.ndarray, min_area_thresh: float, mode: str = "islands"
    ):
        """
        Removes regions that are either too small (below area_thresh) or too big (above max_area_thresh)
        in a mask. Returns the processed mask based on the provided mode.

        Parameters:
            mask (np.ndarray): Binary mask to be processed.
            area_thresh (float): Minimum area threshold. Regions smaller than this are removed.
            max_area_thresh (float): Maximum area threshold. Regions larger than this are removed.
            mode (str): Either "holes" or "islands".
                        - "holes": Remove holes (small or too big holes will be filled).
                        - "islands": Remove islands (keep only regions not removed).

        Returns:
            filtered_mask (np.ndarray): Processed mask based on the mode.
        """
        if mask.dtype != np.bool:
            mask = mask > 0.5
        correct_holes = mode == "holes"
        working_mask = (correct_holes ^ mask).astype(np.uint8)
        n_labels, regions, stats, _ = cv2.connectedComponentsWithStats(working_mask, 8)
        sizes = stats[:, -1][1:]  # Row 0 is background label
        # Identify regions that are too small or too big.
        remove_regions = [i + 1 for i, s in enumerate(sizes) if s < min_area_thresh]
        if len(remove_regions) == 0:
            return mask
        if correct_holes:
            fill_labels = [0] + remove_regions
        else:
            fill_labels = [
                i for i in range(n_labels) if i not in ([0] + remove_regions)
            ]
            # If every region is removed, keep the largest region
            if len(fill_labels) == 0:
                fill_labels = [int(np.argmax(sizes)) + 1]
        mask = np.isin(regions, fill_labels)
        return mask


# Example usage:
if __name__ == "__main__":
    # Provide the image path or use any numpy array representing an image.
    path = (
        "/home/USER/data/arti4d/raw/SCENE/SEQ/RGB"
    )
    # list all the images in the directory

    imgs_path = glob.glob(os.path.join(path, "*.jpg"))
    imgs_path = sorted(imgs_path)

    segmentor = HandSegmentor(resize=(256, 256))
    # seq = segmentor.find_hand_sequence(imgs_path)
    # print(f"Found {len(seq)} hand sequences.")

    # buffer_frames = 20
    # for stf, sof in seq:
    #     for i in range(stf, sof + buffer_frames):
    #         img_path = imgs_path[i]
    #         image = Image.open(img_path)
    #         mask = segmentor.segment(image)
    #         segmentor.visualize(image, mask)
    #         cv2.waitKey(200)
    #     cv2.destroyAllWindows()

    for i in range(120, len(imgs_path), 2):
        img_path = imgs_path[i]
        image = Image.open(img_path).convert("RGB")
        hand_mask = segmentor.segment(image)
        hand_mask = segmentor.remove_small_regions(hand_mask, 1000.0, mode="islands")
        if np.sum(hand_mask) < 150.0:
            loguru.logger.warning(
                f"Hand mask sum is {np.sum(hand_mask):.3f}. Skipping."
            )
            continue
        hand_indices = np.argwhere(hand_mask > 0.99)
        points = segmentor.sample_points_near_hand(
            hand_indices, 10, 100.0
        )  # this points based on the resized hand mask
        segmentor.visualize(image, hand_mask, points=points, alpha=0.5)
        cv2.waitKey(100)
    cv2.destroyAllWindows()
