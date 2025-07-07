# OWL-ViT Pipeline

A reusable framework for zero-shot OWL-ViT inference on high-resolution satellite imagery, designed for modular experimentation and integration into geospatial detection tasks.

---

## 🔧 Features

* Zero-shot detection with [OWL-ViT](https://huggingface.co/google/owlvit-base-patch32)
* Modular experiment design (baseline, tiling, prompt engineering, etc.)
* Visualisation and GeoJSON export of detections
* Support for satellite imagery and archaeological feature prompts
* Easily extensible for your own geospatial detection projects

---

## 🛆 Installation

```bash
git clone https://github.com/yourusername/owlvit-pipeline.git
cd owlvit-pipeline
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

---

## 🌍 Step 1: Download Satellite Image

Use the provided utility to fetch high-resolution satellite imagery.

```bash
python fetch_satellite.py \
  --min-lat 19.6890 --min-lon -98.8550 \
  --max-lat 19.7050 --max-lon -98.8350 \
  --zoom 17 \
  --out data/images/<you_image_file.png>
```

OR Upload your image to the 'data/images' folder
---

## 🚀 Step 2: Run Baseline Inference

```python
from tools.owlvit_utils import OwlViTPipeline

# Initialize pipeline
pipeline = OwlViTPipeline(
    experiment_name="baseline",
    resize_size=(1024, 1024)
)

# Load data
pipeline.load_image("images/teotihuacan_highres.png", image_name="teotihuacan")
pipeline.load_prompts("prompts.txt")

# Run inference
results = pipeline.run_single_experiment(threshold=0.002)

# Save outputs
pipeline.save_visualisation(results, threshold=0.002)
pipeline.save_metrics(results, threshold=0.002)
pipeline.save_geojson(results, threshold=0.002)
```

---

## 📓 Experiments

Each folder under `experiments/` contains a Jupyter notebook implementing a distinct method to improve OWL-ViT zero-shot object detection on high-resolution satellite imagery. These experiments are designed to benchmark performance across various threshold values and inference strategies:

---

### **Baseline**  
`experiments/baseline/notebooks/notebook.ipynb`  
Performs a single-shot full-image inference using OWL-ViT with the chosen prompt set. The raw outputs from the model are saved and visualized across different thresholds. Detection metrics such as total detections, average confidence scores, and per-class counts are logged. This serves as the control experiment to compare all other enhancements.

### **Tiling**  
`experiments/tiling/notebooks/notebook.ipynb`  
Divides the high-resolution satellite image into overlapping tiles (e.g., 768×768 with 0.5 overlap). OWL-ViT is run separately on each tile, and results are merged into a single detection set using coordinate translation and deduplication. This helps capture small-scale archaeological features that are often lost in large-scale inference due to resolution constraints. Metrics and visualizations are again saved across threshold levels.

### **Prompt Engineering**  
`experiments/prompt_engineering/notebooks/notebook.ipynb`  
Tests the effect of prompt phrasing on detection results. For example:
- "an archaeological site in a satellite image"
- "an ancient temple ruin"
- "a stepped tank from above"
The notebook runs inference separately with each prompt or combination of prompts, compares their effectiveness, and analyzes prompt sensitivity. This experiment helps evaluate how open-vocabulary models like OWL-ViT respond to prompt granularity and semantic variation.

### **Pre-processing**  
`experiments/preprocessing/notebooks/notebook.ipynb`  
Applies basic image enhancement techniques to improve visibility before inference:
- **Contrast Stretching**: Linearly scales pixel values to enhance faded or low-contrast features.
- **Histogram Equalization**: Redistributes pixel intensities to accentuate edges and local structures.
These pre-processed images are then passed into the standard full-image OWL-ViT pipeline. Visualizations and detection counts are compared with those from the unprocessed baseline.

### **Post-processing**  
`experiments/postprocessing/notebooks/notebook.ipynb`  
Refines the raw model outputs using the following filtering steps:
1. **Prompt Filtering**: Keeps only detections corresponding to a specific prompt (e.g., archaeological site).
2. **Non-Maximum Suppression (NMS)**: Removes overlapping boxes based on IoU (default: 0.3) to eliminate duplicates.
3. **Area Filtering**: Retains only boxes within a size range (e.g., 0.005%–5% of the full image area).
4. **Top-K Selection**: Selects the top K highest-confidence detections (default: K=12).

Each step reduces noise and focuses on more reliable detections. The final post-filtered boxes are visualized and analyzed.

---

## 🧪 Customisation

This pipeline is modular. You can:

* Add new experiment types by copying a template notebook
* Extend or replace utility scripts under `/tools`
* Swap in new imagery or prompt sets

---

## ✅ Requirements

See `requirements.txt`. Key packages include:

* `transformers`
* `torch`
* `Pillow`
* `matplotlib`
* `geopandas`
* `shapely`
* `rasterio` 
* `scikit-learn`

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).
