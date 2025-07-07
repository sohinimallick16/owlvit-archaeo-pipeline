import os
import glob
import json
import pandas as pd
import geopandas as gpd
from shapely.geometry import box

class EvaluationUtils:
    """
    Helper methods for evaluating OWL-ViT detections against ground truth.
    """

    @staticmethod
    def compute_iou(geom1, geom2):
        """Compute intersection-over-union of two shapely geometries."""
        inter = geom1.intersection(geom2).area
        union = geom1.union(geom2).area
        return inter / union if union > 0 else 0

    @staticmethod
    def load_ground_truth(gt_path: str):
        """
        Load ground truth GeoJSON, ensure 'class' and optional 'subclass' exist, and compute bounding boxes.
        """
        import geopandas as gpd
        from shapely.geometry import box

        gt = gpd.read_file(gt_path)
        # must have 'class' column
        if 'class' not in gt.columns:
            raise ValueError("Ground truth must have a 'class' property.")
        # add bbox geometry for IoU tests
        gt['bbox'] = gt.geometry.bounds.apply(
            lambda r: box(r.minx, r.miny, r.maxx, r.maxy), axis=1
        )
        return gt

    @staticmethod
    def evaluate_threshold(
        method: str,
        threshold: float,
        gt_gdf,
        prompt_list: list,
        prompt_to_class: dict,
        base_dir: str = 'experiments'
    ) -> dict:
        """
        Evaluate a single experiment at one threshold.

        Returns a dict of metrics including localization precision/recall,
        classification accuracy, and proxy counts.
        """
        import os, geopandas as gpd

        # load predictions
        geojson_path = os.path.join(
            base_dir, method, 'outputs', 'geojson',
            f'detections_{method}_teotihuacan_{threshold:.4f}.geojson'
        )
        if not os.path.exists(geojson_path):
            raise FileNotFoundError(f"No GeoJSON for {method}@{threshold}")
        pred = gpd.read_file(geojson_path)
        # map prompt index to class
        pred['pred_class'] = pred['label'].apply(lambda i: prompt_to_class[prompt_list[i]])
        # rename score column if needed
        if 'scores' in pred.columns:
            pred.rename(columns={'scores':'score'}, inplace=True)

        # run detection evaluation
        ed = EvaluationUtils.evaluate_detection(pred, gt_gdf, prompt_list, prompt_to_class, iou_thresh=0.5)
        # add experiment metadata
        ed.update({
            'method': method,
            'threshold': threshold
        })
        return ed

    @staticmethod
    def evaluate_detection(
        pred_gdf,
        gt_gdf,
        prompt_list,
        prompt_to_class,
        iou_thresh: float = 0.5
    ) -> dict:
        """
        Core evaluation: compute localization precision/recall and classification accuracy.
        """
        tp_loc = tp_cls = fp = 0
        matched = set()
        for _, p in pred_gdf.iterrows():
            best_iou = 0; best_j = None
            for j, g in gt_gdf.iterrows():
                iou = EvaluationUtils.compute_iou(p.geometry, g['bbox'])
                if iou > best_iou:
                    best_iou, best_j = iou, j
            if best_iou >= iou_thresh:
                tp_loc += 1
                matched.add(best_j)
                # classification check
                pred_cls = p['pred_class']
                gt_cls   = gt_gdf.loc[best_j, 'class']
                gt_sub   = gt_gdf.loc[best_j].get('subclass', None)
                if pred_cls == gt_sub or pred_cls == gt_cls:
                    tp_cls += 1
            else:
                fp += 1
        fn = len(gt_gdf) - len(matched)
        total = len(pred_gdf)
        avg_score = float(pred_gdf['score'].mean()) if total>0 else 0.0
        arch_count = (pred_gdf['pred_class'] == 'archaeological_site').sum()
        precision_loc = tp_loc/(tp_loc+fp) if tp_loc+fp>0 else 0.0
        recall_loc    = tp_loc/(tp_loc+fn) if tp_loc+fn>0 else 0.0
        class_acc     = tp_cls/tp_loc       if tp_loc>0 else 0.0
        return {
            'total_detections': total,
            'avg_score': avg_score,
            'arch_site_count': int(arch_count),
            'precision_loc': precision_loc,
            'recall_loc': recall_loc,
            'class_acc': class_acc
        }
    """
    Helper methods for evaluating OWL-ViT detections against ground truth.
    """

    @staticmethod
    def compute_iou(geom1, geom2):
        """Compute intersection-over-union of two shapely geometries."""
        inter = geom1.intersection(geom2).area
        union = geom1.union(geom2).area
        return inter / union if union > 0 else 0

    @staticmethod
    def load_ground_truth(gt_path: str) -> gpd.GeoDataFrame:
        """Load ground truth GeoJSON and convert polygons to bounding boxes."""
        gt = gpd.read_file(gt_path)
        # ensure class column exists
        if 'class' not in gt.columns:
            raise ValueError("Ground truth GeoJSON must have a 'class' property on each feature.")
        # add bbox geometry
        gt['bbox'] = gt.geometry.bounds.apply(
            lambda row: box(row.minx, row.miny, row.maxx, row.maxy), axis=1
        )
        return gt

    @staticmethod
    def evaluate_detection(
        pred_gdf: gpd.GeoDataFrame,
        gt_gdf: gpd.GeoDataFrame,
        prompt_list: list[str],
        prompt_to_class: dict[str,str],
        iou_thresh: float = 0.5
    ) -> dict:
        """
        Evaluate a single set of detections against ground truth.

        Returns precision_loc, recall_loc, class_acc, total_detections, avg_score, arch_site_count
        """
        tp_loc = 0
        tp_cls = 0
        fp = 0
        matched_gt = set()

        for _, p in pred_gdf.iterrows():
            best_iou = 0
            best_j = None
            for j, g in gt_gdf.iterrows():
                iou = EvaluationUtils.compute_iou(p.geometry, g['bbox'])
                if iou > best_iou:
                    best_iou = iou
                    best_j = j
            if best_iou >= iou_thresh:
                tp_loc += 1
                matched_gt.add(best_j)
                # classification check
                prompt = prompt_list[p['label']]
                pred_class = prompt_to_class.get(prompt, None)
                if pred_class == gt_gdf.loc[best_j, 'class']:
                    tp_cls += 1
            else:
                fp += 1

        fn = len(gt_gdf) - len(matched_gt)
        precision_loc = tp_loc / (tp_loc + fp) if tp_loc + fp > 0 else 0
        recall_loc    = tp_loc / (tp_loc + fn) if tp_loc + fn > 0 else 0
        class_acc     = tp_cls / tp_loc       if tp_loc > 0 else 0

        total = len(pred_gdf)
        avg_score = float(pred_gdf['score'].mean()) if total > 0 else 0
        # count arch_site_count
        arch_count = (pred_gdf['pred_class'] == 'archaeological_site').sum()

        return {
            'precision_loc': precision_loc,
            'recall_loc': recall_loc,
            'class_acc': class_acc,
            'total_detections': total,
            'avg_score': avg_score,
            'arch_site_count': int(arch_count)
        }

    @staticmethod
    def aggregate_results(
        experiments: list[str],
        thresholds: list[float],
        gt_path: str,
        prompt_list: list[str],
        prompt_to_class: dict[str,str],
        base_dir: str = 'experiments'
    ) -> pd.DataFrame:
        """
        Loop over experiments and thresholds, evaluate and return a DataFrame.
        """
        gt = EvaluationUtils.load_ground_truth(gt_path)
        records = []
        for method in experiments:
            for thr in thresholds:
                geo_path = os.path.join(
                    base_dir, method, 'outputs', 'geojson',
                    f'detections_{method}_teotihuacan_{thr:.4f}.geojson'
                )
                if not os.path.exists(geo_path):
                    continue
                pred = gpd.read_file(geo_path)
                # map labels to classes
                pred['pred_class'] = pred['label'].apply(lambda i: prompt_to_class[prompt_list[i]])
                # ensure score column exists
                pred.rename(columns={'scores':'score'}, inplace=True)
                metrics = EvaluationUtils.evaluate_detection(
                    pred, gt, prompt_list, prompt_to_class, iou_thresh=thr
                )
                metrics.update({ 'method': method, 'threshold': thr })
                records.append(metrics)
        df = pd.DataFrame(records)
        return df
