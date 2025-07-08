import os
import json, glob
from typing import List, Dict, Any, Optional

from PIL import Image
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pandas as pd
from IPython.display import display, HTML
from shapely.geometry import box as shapely_box
import geopandas as gpd
from transformers import OwlViTProcessor, OwlViTForObjectDetection


class OwlViTPipeline:
    """
    A configurable pipeline for running OWL-ViT object detection on satellite imagery,
    for geospatial archaeology.

    This class centralizes data path management, model loading, inference loops,
    visualization, metric collection, and GeoJSON export. It is designed
    to be reused across multiple experiments or projects with minimal code changes.

    Attributes:
        data_dir (str): Directory containing raw inputs (images, prompt files).
        baseline_dir (str): Base folder for storing experiment outputs.
        output_dir (str): Parent directory for all output files (PNGs, JSONs).
        visualisations_dir (str): Directory for saved detection visualizations.
        geojson_dir (str): Directory for saved GeoJSON exports.
        processor (OwlViTProcessor): Pretrained text-image processor.
        model (OwlViTForObjectDetection): Pretrained object detection model.
        resize_size (tuple): Target (width, height) for image resizing.
        image_full (Image.Image): Loaded full-resolution PIL image.
        image_resized (Image.Image): Resized PIL image used for inference.
        prompts (List[str]): List of textual prompts for detection.
    """

    def __init__(
        self,
        experiment_name: str,
        root_level: int = 2,
        model_name: str = 'google/owlvit-base-patch32',
        resize_size: tuple = (1024, 1024)
    ):
        """
        Initialize paths, load the OWL-ViT model, and set resize parameters.

        Args:
            root_level (int): Number of parent directories up to project root.
            baseline_subdir (str): Name of subdirectory for baseline experiments.
            model_name (str): HF identifier of the OWL-ViT model to load.
            resize_size (tuple): (width, height) to which input images are resized.
        """
        # Paths
        
        # --- project root ---
        self.ROOT = os.path.abspath(
            os.path.join(__file__, *([".."]*root_level))
        )
        # --- experiment subfolder  ---
        self.experiment = experiment_name
        self.experiment_dir = os.path.join(self.ROOT, "experiments", experiment_name)

        # --- outputs ---
        self.data_dir            = os.path.join(self.ROOT, "data")
        self.output_dir          = os.path.join(self.experiment_dir, "outputs")
        self.visualisations_dir  = os.path.join(self.output_dir, "visualisations")
        self.geojson_dir         = os.path.join(self.output_dir, "geojson")
        self.metrics_dir         = os.path.join(self.output_dir, "metrics")
       
        # ensure that all output directories exist
        for d in (
            self.experiment_dir,
            self.output_dir,
            self.visualisations_dir,
            self.geojson_dir,
            self.metrics_dir,
        ):
            os.makedirs(d, exist_ok=True)

        # Model
        self.processor = OwlViTProcessor.from_pretrained(model_name)
        self.model = OwlViTForObjectDetection.from_pretrained(model_name)

        # Resize config
        self.resize_size = resize_size

    def load_image(self, filename: str, image_name: Optional[str] = None) -> None:
        """
        Load a high-resolution image and its resized copy for inference.

        Args:
            filename (str): Relative path (under data_dir) of the image file.

        Effects:
            Sets `self.image_full` and `self.image_resized`.

        Raises:
            FileNotFoundError: If the image file does not exist.

        Example:
            pipeline.load_image('images/my_satellite.png')
        """
        path = os.path.join(self.data_dir, 'images', filename)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Image not found: {path}")
        self.image_full = Image.open(path).convert("RGB")
        self.image_resized = self.image_full.resize(self.resize_size)
        # store a base name to use in all output filenames
        self.image_name = image_name or os.path.splitext(filename)[0]

    def load_prompts(self, filename: str) -> None:
        """
        Read detection prompts from a text file, one prompt per line.

        Args:
            filename (str): Relative path (under data_dir) for prompts.txt.

        Effects:
            Populates `self.prompts` as a list of strings.

        Raises:
            FileNotFoundError: If the prompt file does not exist.

        Example:
            pipeline.load_prompts('prompts.txt')
        """
        path = os.path.join(self.data_dir, filename)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Prompts not found: {path}")
        with open(path, 'r') as f:
            self.prompts = [line.strip() for line in f if line.strip()]

    def run_single_experiment(
        self,
        threshold: float
    ) -> Dict[str, Any]:
        """
        Convenience wrapper to run detection at a single threshold.

        Args:
            threshold (float): Confidence threshold for post-processing.

        Returns:
            Dict[str, Any]: Metric dict for the given threshold.

        Example:
            >>> metric = pipeline.run_single_experiment(0.002)
        """
        # call the multi-threshold method with a single-element list
        metrics = self.run_threshold_experiments([threshold])
        return metrics[0]
    
    def run_threshold_experiments(
        self,
        thresholds: List[float]
    ) -> List[Dict[str, Any]]:
        """
        Execute detection at multiple score thresholds, save visual outputs and metrics.

        Args:
            thresholds (List[float]): Confidence thresholds for post-processing.

        Returns:
            List[Dict[str, Any]]: List of metric dictionaries with keys:
                - 'threshold': float
                - 'total_detections': int
                - 'avg_score': float
                - 'per_class_counts': Dict[str,int]

        Effects:
            * Saves PNG visualisations in `visualisations_dir`.
            * Writes `detection_metrics.json` in `output_dir`.

        Example:
            metrics = pipeline.run_threshold_experiments([0.01, 0.005])
        """
        metrics = []
        target_size = torch.tensor([[
            self.image_full.height,
            self.image_full.width
        ]])

        for thresh in thresholds:
            inputs = self.processor(
                text=self.prompts,
                images=self.image_resized,
                return_tensors="pt"
            )
            with torch.no_grad():
                outputs = self.model(**inputs)
            results = self.processor.post_process_object_detection(
                outputs,
                target_sizes=target_size,
                threshold=thresh
            )[0]

            # Collect metrics
            total = len(results['scores'])
            avg_score = results['scores'].mean().item() if total else 0.0
            counts = {}
            for lbl in results['labels'].tolist():
                name = self.prompts[lbl]
                counts[name] = counts.get(name, 0) + 1
            metrics.append({
                'threshold': float(thresh),
                'total_detections': total,
                'avg_score': avg_score,
                'per_class_counts': counts
            })
            
            # Save visualization and metrics using helpers
            self.save_visualisation(results, threshold=thresh)
            self.save_metrics     (results, threshold=thresh)
            
        return metrics
    
    def save_visualisation(
        self,
        results: Dict[str, Any],
        threshold: float,
    ) -> str:
        """
        Draw `results` on the full image and save a PNG.

        Returns:
            The filepath of the saved PNG.
        """
        fig, ax = plt.subplots(figsize=(12, 12))
        ax.imshow(self.image_full)
        for box in results['boxes']:
            x1, y1, x2, y2 = box.tolist()
            rect = patches.Rectangle(
                (x1, y1), x2 - x1, y2 - y1,
                linewidth=2, edgecolor='red', facecolor='none'
            )
            ax.add_patch(rect)
        ax.axis('off')

        name = f"detections_{self.experiment}_{self.image_name}_{threshold:.4f}.png"

        path = os.path.join(self.visualisations_dir, name)
        os.makedirs(self.visualisations_dir, exist_ok=True)
        fig.savefig(path)
        plt.close(fig)
        return path

    def save_metrics(
        self,
        results: Dict[str, Any],
        threshold: float,
        suffix: str = ''
    ) -> str:
        """
        Compute and save a metrics JSON for `results`.

        Returns:
            The filepath of the saved JSON.
        """
        counts = {}
        for lbl in results['labels'].tolist():
            key = self.prompts[lbl]
            counts[key] = counts.get(key, 0) + 1

        metric = {
            'threshold': float(threshold),
            'total_detections': len(results['scores']),
            'avg_score': float(results['scores'].mean()),
            'per_class_counts': counts
        }

        name = f"detections_{self.experiment}_{self.image_name}"
        if suffix:
            name += f"_{suffix}"
        name += f"_{threshold:.4f}.json"

        path = os.path.join(self.metrics_dir, name)
        os.makedirs(self.metrics_dir, exist_ok=True)
        with open(path, 'w') as f:
            json.dump(metric, f, indent=2)
        return path
    
    def display_metrics(self) -> None:
        """
        Read all per-threshold metrics JSONs in `metrics_dir`, build a DataFrame,
        and render it as a styled HTML table in a Jupyter notebook.
        """

        pattern = os.path.join(
            self.metrics_dir,
            f"detections_{self.experiment}_{self.image_name}_*.json"
        )
        records = []
        for fp in sorted(glob.glob(pattern)):
            with open(fp) as f:
                records.append(json.load(f))
        if not records:
            print("No metrics files found.")
            return

        df = pd.DataFrame(records)
        # reorder columns
        cols = ["threshold", "total_detections", "avg_score"] + \
            [c for c in df.columns if c not in ("threshold", "total_detections", "avg_score")]
        df = df[cols]
        html_table = df.to_html(
            index=False,
            border=0,
            classes=["table", "table-striped", "table-hover"],
            float_format="%.3f"
        )
        display(HTML(html_table))


    def save_geojson(
        self,
        results_geo: Dict[str, Any],
        threshold: float
    ) -> str:
        """
        Export post-processed detections to a GeoJSON file.

        Args:
            results_geo (Dict[str, Any]): Single-threshold output from `post_process_object_detection`.
            threshold (float): Corresponding confidence threshold.

        Returns:
            str: Filepath of the saved GeoJSON.

        Example:
            path = pipeline.save_geojson(results, 0.005)
        """
        features = []
        for score, label, box in zip(
            results_geo['scores'], results_geo['labels'], results_geo['boxes']
        ):
            xmin, ymin, xmax, ymax = box.tolist()
            geom = shapely_box(xmin, ymin, xmax, ymax)
            features.append({
                'geometry': geom,
                'score': float(score),
                'label': self.prompts[label]
            })
        gdf = gpd.GeoDataFrame(features, crs="EPSG:4326")
        fname = f'detections_{self.experiment}_{self.image_name}_{threshold:.4f}.geojson'
        path = os.path.join(self.geojson_dir, fname)
        os.makedirs(self.geojson_dir, exist_ok=True)
        gdf.to_file(path, driver='GeoJSON')
        return path

    def show_only_prompt(
        self,
        results: Dict[str, Any],
        target_prompt: str
    ) -> None:
        """
        Visually isolate and display detections for a single prompt.

        Args:
            results (Dict[str, Any]): Output dict from one call to `post_process_object_detection`.
            target_prompt (str): Exact prompt string to filter.

        Raises:
            ValueError: If `target_prompt` is not in the loaded prompts list.

        Example:
            pipeline.show_only_prompt(results, 'archaeological site')
        """
        if target_prompt not in self.prompts:
            raise ValueError(f"Prompt '{target_prompt}' not found.")
        idx = self.prompts.index(target_prompt)
        boxes, scores, labels = (
            results['boxes'], results['scores'], results['labels']
        )
        keep = [i for i, lbl in enumerate(labels.tolist()) if lbl == idx]
        if not keep:
            print(f"No detections for '{target_prompt}'")
            return

        fig, ax = plt.subplots(figsize=(8, 8))
        ax.imshow(self.image_full)
        for i in keep:
            xmin, ymin, xmax, ymax = boxes[i].tolist()
            score = scores[i].item()
            rect = patches.Rectangle(
                (xmin, ymin), xmax-xmin, ymax-ymin,
                linewidth=2, edgecolor='red', facecolor='none'
            )
            ax.add_patch(rect)
            ax.text(
                xmin, ymin - 5,
                f"{target_prompt} ({score:.2f})",
                color='red', fontsize=12
            )
        ax.axis('off')
        plt.show()
        
    def run_and_show_prompt(self, threshold: float, target_prompt: str) -> None:
        """
        Convenience method: run inference+postprocess at `threshold`, then
        filter and plot only `target_prompt` in one call.
        """
        # inference + postprocessing
        inputs = self.processor(text=self.prompts, images=self.image_resized, return_tensors="pt")
        with torch.no_grad():
            outputs = self.model(**inputs)
        target_size = torch.tensor([[self.image_full.height, self.image_full.width]])
        results = self.processor.post_process_object_detection(
            outputs, target_sizes=target_size, threshold=threshold
        )[0]
        # delegate to existing plotter
        self.show_only_prompt(results, target_prompt)
        
    def run_and_save_geojson(self, threshold: float) -> str:
        """
        Convenience method: run inference + postprocess at `threshold`, then save all detections
        as a GeoJSON under the experiment's geojson directory.

        Args:
            threshold (float): Confidence threshold for post-processing.

        Returns:
            str: Filepath of the saved GeoJSON.

        Example:
            path = pipeline.run_and_save_geojson(0.003)
        """
        # inference + postprocessing
        inputs = self.processor(text=self.prompts, images=self.image_resized, return_tensors="pt")
        with torch.no_grad():
            outputs = self.model(**inputs)
        target_size = torch.tensor([[self.image_full.height, self.image_full.width]])
        results_geo = self.processor.post_process_object_detection(
            outputs, target_sizes=target_size, threshold=threshold
        )[0]
        # save via existing helper
        return self.save_geojson(results_geo, threshold)
        