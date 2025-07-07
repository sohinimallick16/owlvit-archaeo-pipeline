# tiling_utils.py

import torch
from PIL import Image
from transformers import OwlViTProcessor, OwlViTForObjectDetection
from typing import List, Dict, Any, Optional

class TilingUtils:
    """
    Helper class for tiled inference of OWL-ViT.
    """

    def __init__(self, processor: OwlViTProcessor, model: OwlViTForObjectDetection, resize_size=(1024,1024)):
        """
        Args:
            processor: an initialized OwlViTProcessor
            model: an initialized OwlViTForObjectDetection
        """
        self.processor = processor
        self.model = model
        self.resize_size = resize_size

    @staticmethod
    def sliding_window(
        image: Image.Image,
        tile_size: int = 1024,
        overlap: float = 0.2
    ):
        """
        Yield (tile_img, x_off, y_off) for tiling a large image.
        """
        w, h = image.size
        step = int(tile_size * (1 - overlap))
        for y in range(0, h, step):
            for x in range(0, w, step):
                x0 = min(x, w - tile_size)
                y0 = min(y, h - tile_size)
                box = (x0, y0, x0 + tile_size, y0 + tile_size)
                yield image.crop(box), x0, y0

    def run_tiled_inference(
            self,
            prompts: list,
            image: Image.Image,
            tile_size: int = 1024,
            overlap: float = 0.2,
            threshold: float = 0.002
        ) -> Dict[str, torch.Tensor]:
            """
            Run OWL-ViT on overlapping tiles and aggregate detections.

            Each crop is resized up to self.resize_size before inference,
            then post-processed back to the original tile dimensions.

            Args:
                prompts (List[str]): Text prompts.
                image (PIL.Image): Full-resolution input.
                tile_size (int): Size of each square crop.
                overlap (float): Fractional overlap between tiles (0–1).
                threshold (float): Confidence threshold.

            Returns:
                Dict[str, torch.Tensor]: keys 'boxes', 'scores', 'labels'
                    aggregated in full-image coordinates.
            """
            all_boxes, all_scores, all_labels = [], [], []

            # Slide a window over the full image
            step = int(tile_size * (1 - overlap))
            w, h = image.size
            for y in range(0, h, step):
                for x in range(0, w, step):
                    x0 = min(x, w - tile_size)
                    y0 = min(y, h - tile_size)
                    tile = image.crop((x0, y0, x0 + tile_size, y0 + tile_size))

                    # Resize each tile to the pipeline's target size
                    tile_resized = tile.resize(self.resize_size)

                    # Run the model
                    inputs = self.processor(text=prompts, images=tile_resized, return_tensors="pt")
                    with torch.no_grad():
                        outputs = self.model(**inputs)

                    # Post-process back to the original tile dimensions
                    results = self.processor.post_process_object_detection(
                        outputs,
                        target_sizes=torch.tensor([[tile.size[1], tile.size[0]]]),
                        threshold=threshold
                    )[0]

                    # Shift boxes into global coords
                    for box, score, lbl in zip(results["boxes"], results["scores"], results["labels"]):
                        xmin, ymin, xmax, ymax = box.tolist()
                        all_boxes.append([xmin + x0, ymin + y0, xmax + x0, ymax + y0])
                        all_scores.append(score.item())
                        all_labels.append(lbl.item())

            return {
                "boxes": torch.tensor(all_boxes),
                "scores": torch.tensor(all_scores),
                "labels": torch.tensor(all_labels)
            }

