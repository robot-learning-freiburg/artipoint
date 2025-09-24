import torch
import numpy as np
from mobile_sam import SamAutomaticMaskGenerator, SamPredictor, sam_model_registry
import cv2
from PIL import Image
import loguru


class MobileSAMSegmenter:
    def __init__(self, config: dict):
        """
        Initialize the MobileSAM model using a configuration dictionary.

        :param config: Configuration dictionary containing 'checkpoint', 'model_type', and 'device'.
        """
        self.device = config.get(
            "device", torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.model = sam_model_registry[config["model_type"]](
            checkpoint=config["checkpoint"]
        )
        self.model.to(self.device)
        self.model.eval()
        self.mask_generator = SamAutomaticMaskGenerator(
            self.model,
            points_per_side=config.get("points_per_side", 20),
            points_per_batch=config.get("points_per_batch", 128),
            pred_iou_thresh=config.get("pred_iou_thresh", 0.5),
            stability_score_thresh=config.get("stability_score_thresh", 0.92),
            stability_score_offset=config.get("stability_score_offset", 0.7),
        )
        self.predictor = SamPredictor(self.model)

    def _resize_image(self, image: Image.Image | np.ndarray, input_size: int):
        """Resize image to maintain aspect ratio with the largest side equal to input_size."""
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image.astype(np.uint8))
        w, h = image.size
        scale = input_size / max(w, h)
        new_size = (int(w * scale), int(h * scale))
        return image.resize(new_size), scale

    def segment_everything(self, image: Image.Image, input_size: int = 1024):
        """
        Automatically segment all objects in the image.

        :param image: PIL Image.
        :param input_size: Target size for the largest image dimension.
        :return: List of segmentation annotations.
        """
        resized, _ = self._resize_image(image, input_size)
        annotations = self.mask_generator.generate(np.array(resized))
        masks = np.array([ann["segmentation"] for ann in annotations])
        scores = np.array([ann["stability_score"] for ann in annotations])
        return masks, scores, None

    def segment_object_with_points(
        self,
        image: Image.Image,
        points: list,
        point_labels: list,
        input_size: int = 1024,
        multimask_output: bool = True,
    ):
        """
        Segment objects in the image guided by user-provided points.

        :param image: PIL Image.
        :param points: List of [x, y] coordinates.
        :param point_labels: List of labels (1 for foreground, 0 for background).
        :param input_size: Target size for the largest image dimension.
        :param multimask_output: Whether to return multiple masks.
        :return: (masks, scores, logits) as returned by the SAM predictor.
        """
        resized, scale = self._resize_image(image, input_size)
        # Scale the points to the resized image coordinates.
        pts = np.array([[int(x * scale), int(y * scale)] for x, y in points])
        labels = np.array(point_labels)
        self.predictor.set_image(np.array(resized))
        masks, scores, logits = self.predictor.predict(
            point_coords=pts, point_labels=labels, multimask_output=multimask_output
        )
        return masks, scores, logits

    def segment_multiple_objects_with_points(
        self,
        image: Image.Image,
        points: list,
        point_labels: list,
        input_size: int = 1024,
    ):
        """
        Segments multiple objects in an image based on provided points and their labels.

        Args:
            image (Image.Image): The input image to be segmented.
            points (list): A list of (x, y) tuples representing the coordinates of the points.
            point_labels (list): A list of labels corresponding to each point.
            input_size (int, optional): The size to which the input image should be resized. Default is 1024.

        Returns:
            tuple: A tuple containing:
            - masks (numpy.ndarray): The segmented masks for each object.
            - scores (numpy.ndarray): The confidence scores for each mask.
            - logits (torch.Tensor): The raw logits from the predictor.
        """
        resized, scale = self._resize_image(image, input_size)
        # Convert the points to torch tensors.
        pts_tensor = (
            torch.tensor(points, dtype=torch.float32, device=self.device).view(-1, 1, 2)
            * scale
        )
        labels_tensor = torch.tensor(
            point_labels, dtype=torch.int64, device=self.device
        ).view(-1, 1)
        self.predictor.set_image(np.array(resized))
        masks, scores, logits = self.predictor.predict_torch(pts_tensor, labels_tensor)
        masks = masks.cpu().numpy()
        scores = scores.cpu().numpy()
        return masks, scores, logits

    @staticmethod
    def remove_small_regions(
        masks: list, min_area_thresh: float, mode: str = "islands"
    ):
        """
        Removes regions that are either too small (below area_thresh) or too big (above max_area_thresh)
        in a mask. Returns the processed mask based on the provided mode.

        Parameters:
            masks (list): List of binary masks.
            area_thresh (float): Minimum area threshold. Regions smaller than this are removed.
            max_area_thresh (float): Maximum area threshold. Regions larger than this are removed.
            mode (str): Either "holes" or "islands".
                        - "holes": Remove holes (small or too big holes will be filled).
                        - "islands": Remove islands (keep only regions not removed).

        Returns:
            list: List of filtered masks.
        """
        filtered_masks = []
        for mask in masks:
            if mask.dtype != np.bool:
                mask = mask > 0.5
            correct_holes = mode == "holes"
            working_mask = (correct_holes ^ mask).astype(np.uint8)
            n_labels, regions, stats, _ = cv2.connectedComponentsWithStats(
                working_mask, 8
            )
            sizes = stats[:, -1][1:]  # Row 0 is background label
            # Identify regions that are too small or too big.
            remove_regions = [i + 1 for i, s in enumerate(sizes) if s < min_area_thresh]
            if len(remove_regions) == 0:
                filtered_masks.append(mask)
                continue
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
            filtered_masks.append(mask)
        return filtered_masks

    def filter_masks_by_score(self, masks, scores):
        """
        Filter masks by selecting the highest scoring mask for each object.

        :param masks: Array of masks.
        :param scores: Array of scores corresponding to the masks.
        :return: List of best masks based on scores.
        """
        best_masks = []
        for i in range(masks.shape[0]):
            best_mask = masks[i][np.argmax(scores[i])]
            best_masks.append(best_mask)
        return best_masks

    def filter_masks_by_size(self, masks, min_size=500, max_size=5000):
        """
        Filter masks by size, keeping only those within the specified size range.

        :param masks: List of binary masks.
        :param min_size: Minimum size of the mask to keep.
        :param max_size: Maximum size of the mask to keep.
        :return: List of filtered masks.
        """
        filtered_masks = []
        for mask in masks:
            mask_size = np.sum(mask)
            if min_size <= mask_size <= max_size:
                filtered_masks.append(mask)
        return filtered_masks

    def filter_redeundant_masks(self, masks, iou_thresh=0.75):
        """
        Filter out redundant masks based on their IoU with the highest scoring mask.

        :param masks: List of binary masks.
        :param iou_thresh: IoU threshold to consider two masks as the same object.
        :return: List of filtered masks.
        """
        filtered_masks = []
        for i, mask in enumerate(masks):
            if i == 0:
                filtered_masks.append(mask)
                continue
            iou = self.compute_iou(mask, filtered_masks[0])
            if iou < iou_thresh:
                filtered_masks.append(mask)
        return filtered_masks

    @staticmethod
    def compute_iou(mask1, mask2):
        """
        Compute the Intersection over Union (IoU) between two binary masks.

        :param mask1: First binary mask.
        :param mask2: Second binary mask.
        :return: The IoU value.
        """
        intersection = np.logical_and(mask1, mask2)
        union = np.logical_or(mask1, mask2)
        return np.sum(intersection) / np.sum(union)

    def draw_points(
        self,
        image: Image.Image,
        points: list,
        point_labels: list,
        point_radius: int = 8,
    ):
        """
        Draw points on an image for visualization using OpenCV.

        :param image: PIL Image.
        :param points: List of [x, y] coordinates.
        :param point_labels: List of labels (1 for foreground, 0 for background).
        :param point_radius: Radius of the drawn point.
        :return: The annotated image as a numpy array.
        """
        img = np.array(image)
        # ensure that points are integers
        points = [(int(x), int(y)) for x, y in points]
        for (x, y), label in zip(points, point_labels):
            color = (255, 255, 0) if label == 1 else (255, 0, 255)
            cv2.circle(img, (x, y), point_radius, color, -1)
        return img

    def visualize_segmentation(
        self, image: np.ndarray, masks: np.ndarray, input_size: int = 1024
    ):
        """
        Visualize segmentation annotations on the image using OpenCV.

        :param image: np.ndarray.
        :param masks: Batched output of multiple object masks for the same image.
        :param input_size: Target size for the largest image dimension.
        :return: Image with visualized annotations.
        """
        img, _ = self._resize_image(image, input_size)
        img = np.array(img)
        for mask in masks:
            color = np.random.randint(0, 255, (3,), dtype=np.uint8)
            img[mask] = img[mask] * 0.5 + color * 0.5
            contours, _ = cv2.findContours(
                mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            cv2.drawContours(img, contours, -1, color.tolist(), 2)
        img = cv2.resize(img, (image.shape[1], image.shape[0]))
        return img


# Example usage:
if __name__ == "__main__":
    from PIL import Image
    from pathlib import Path
    from torch.utils.data import DataLoader
    import time
    from azure_dataloader import AzureRGBDDataset

    root_path = Path(
        "home/USER/input_data/arti4d/raw/SCENE/SEQ"
    )
    azure_cfg = {
        "root_dir": "data",
        "transforms": None,
        "depth_min": 0.5,
        "depth_max": 3.0,
        "root_path": root_path,
    }

    dataset = AzureRGBDDataset(azure_cfg)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    print("Dataset length:", len(dataset))

    sam_config = {
        "checkpoint": "/home/USER/artipoint/checkpoints/weight/mobile_sam.pt",
        "model_type": "vit_t",
        "device": "cuda" if torch.cuda.is_available() else "cpu",
    }

    # Initialize with your model configuration.
    seg = MobileSAMSegmenter(config=sam_config)

    p = np.meshgrid(np.linspace(0, 1280, 10), np.linspace(0, 720, 10))
    p = np.stack((p[0].ravel(), p[1].ravel()), axis=1)
    p = [(int(x), int(y)) for x, y in p]

    # p = [(450, 450)]

    # Test Loop
    skip = 2
    for i in range(500, len(dataset), skip):
        img = dataset[i]["rgb"].astype(np.uint8)
        st = time.time()
        masks, scores, _ = seg.segment_multiple_objects_with_points(
            img, points=p, point_labels=[1] * len(p)
        )
        masks = seg.filter_masks_by_score(masks, scores)
        masks = seg.remove_small_regions(masks, 1000)
        masks = seg.filter_masks_by_size(masks, 1000, 5000)
        masks = seg.filter_redeundant_masks(masks, iou_thresh=0.25)
        # anns = seg.segment_everything(img)
        # masks = [ann["segmentation"] for ann in anns]
        # masks = np.expand_dims(np.array(masks), axis=0)
        img = seg.visualize_segmentation(img, masks)
        # img_p = seg.draw_points(img, p, [1] * len(p))
        cv2.imshow("Segmentation", img)
        cv2.waitKey(1)
    cv2.destroyAllWindows()
