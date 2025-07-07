import torch
from torchvision.ops import nms

class PostprocessUtils:
    """
    A collection of post-processing routines for OWL-ViT detections.
    """

    @staticmethod
    def filter_by_prompt(results: dict, prompt: str, prompts: list) -> dict:
        """
        Keep only detections whose label matches `prompt`.

        Args:
            results: dict with 'boxes', 'scores', 'labels'
            prompt: the exact prompt string to filter by
            prompts: list of all prompt strings

        Returns:
            Filtered results dict
        """
        if prompt not in prompts:
            raise ValueError(f"Prompt '{prompt}' not found in prompts list.")
        idx = prompts.index(prompt)
        labels = results['labels']
        mask = (labels == idx)
        return {
            'boxes': results['boxes'][mask],
            'scores': results['scores'][mask],
            'labels': results['labels'][mask]
        }

    @staticmethod
    def apply_nms(results: dict, iou_threshold: float = 0.3) -> dict:
        """
        Apply non-maximum suppression to prune overlapping boxes.

        Args:
            results: dict with 'boxes','scores','labels'
            iou_threshold: float IoU threshold for suppression

        Returns:
            Pruned results dict
        """
        boxes = results['boxes']
        scores = results['scores']
        labels = results['labels']
        keep = nms(boxes, scores, iou_threshold=iou_threshold)
        return {
            'boxes': boxes[keep],
            'scores': scores[keep],
            'labels': labels[keep]
        }

    @staticmethod
    def area_filter(results: dict, min_pct: float = 0.00005, max_pct: float = 0.05, image_size: tuple = None) -> dict:
        """
        Filter boxes by area percentage of the full image.

        Args:
            results: dict with 'boxes','scores','labels'
            min_pct: min area fraction (e.g. 0.00005 for 0.005%)
            max_pct: max area fraction (e.g. 0.05 for 5%)
            image_size: (width, height) of full image

        Returns:
            Filtered results dict
        """
        if image_size is None:
            raise ValueError("image_size must be provided as (width, height)")
        img_w, img_h = image_size
        areas = (results['boxes'][:,2] - results['boxes'][:,0]) * (results['boxes'][:,3] - results['boxes'][:,1])
        img_area = img_w * img_h
        mask = (areas >= min_pct * img_area) & (areas <= max_pct * img_area)
        return {
            'boxes': results['boxes'][mask],
            'scores': results['scores'][mask],
            'labels': results['labels'][mask]
        }

    @staticmethod
    def top_k(results: dict, k: int = 12) -> dict:
        """
        Keep only the top-k highest-scoring detections.

        Args:
            results: dict with 'boxes','scores','labels'
            k: number of top detections to keep

        Returns:
            Pruned results dict
        """
        scores = results['scores']
        if len(scores) <= k:
            return results
        topk = torch.topk(scores, k).indices
        return {
            'boxes': results['boxes'][topk],
            'scores': results['scores'][topk],
            'labels': results['labels'][topk]
        }
