# OWL-ViT Archaeology Inference Pipeline

[![PyPI Version](https://img.shields.io/pypi/v/owlvit-archaeo-pipeline.svg)](https://pypi.org/project/owlvit-archaeo-pipeline)
[![License](https://img.shields.io/github/license/your-org/owlvit-archaeo-pipeline)](LICENSE)

A lightweight, installable toolkit for running and benchmarking zero-shot OWL-ViT inference on geospatial imagery. Supports data acquisition, tiling, prompt engineering, pre- and post-processing, and automated benchmarking.

---

## Step 1: Fetch Satellite Imagery

Use the provided `fetch_satellite.py` script to download and stitch ESRI World Imagery tiles for your area of interest without QGIS.

```bash
python fetch_satellite.py \
  --min-lat <MIN_LAT> --min-lon <MIN_LON> \
  --max-lat <MAX_LAT> --max-lon <MAX_LON> \
  --zoom <ZOOM_LEVEL> \
  --out <OUTPUT_FILENAME>.png
```

Example:

```bash
python fetch_satellite.py \
  --min-lat 19.692 --min-lon -98.860 \
  --max-lat 19.722 --max-lon -98.820 \
  --zoom 18 \
  --out teotihuacan.png
```

This will save the stitched image under `data/images/<OUTPUT_FILENAME>.png` by default.

---

## Table of Contents

* [Features](#features)
* [Installation](#installation)
* [Project Structure](#project-structure)
* [Quick Start](#quick-start)
* [API Reference](#api-reference)
* [Running Benchmarks](#running-benchmarks)
* [Contributing](#contributing)
* [License](#license)

---

## Features

* **Data Acquisition**: Download and stitch high-resolution satellite imagery via ESRI World Imagery tiles
* **Zero-shot OWL-ViT inference** on high-resolution geospatial imagery
* **Flexible tiling strategies** for sliding-window detection
* **Prompt engineering utilities** for open-vocabulary retrieval
* **Pre- and post-processing helpers** (thresholding, non-max suppression)
* **Automated benchmarking** across multiple configurations

---

## Installation

```bash
git clone https://github.com/your-org/owlvit-archaeo-pipeline.git
cd owlvit-archaeo-pipeline
pip install -e .
```

Ensure you have Python 3.8+ and a GPU-enabled PyTorch installation.

---

## Project Structure

```
owlvit-archaeo-pipeline/
├─ fetch_satellite.py       # Step 1: download & stitch imagery
├─ tools/
│   ├─ owlvit_utils.py      # OwlViTPipeline class for inference & outputs
│   └─ tiling_utils.py      # TilingUtils for sliding-window inference
│
├─ experiments/             # Example notebooks for each method
│   ├─ baseline/
│   │   └─ notebook.ipynb
│   └─ prompt_engineering/
│       └─ notebook.ipynb
│
├─ run_benchmark.py         # Execute notebooks & aggregate results
├─ setup.py                 # Package definition for pip
├─ requirements.txt         # Dependencies
└─ README.md                # This guide
```

---

## Quick Start

1. **Fetch imagery** (see [Step 1](#step-1-fetch-satellite-imagery)).
2. **Install the package**

   ```bash
   pip install -e .
   ```
3. **Prepare prompts & run inference**

   ```python
   from tools.owlvit_utils import OwlViTPipeline

   pipeline = OwlViTPipeline(experiment_name='baseline')
   pipeline.load_image('data/images/teotihuacan.png', image_name='teotihuacan')
   pipeline.load_prompts('data/prompts.txt')

   thresholds = [0.002, 0.005, 0.01]
   results = pipeline.run_threshold_experiments(thresholds)

   pipeline.save_visualisation(results[0], thresholds[0], out_path='outputs/vis_teotihuacan_0.002.png')
   pipeline.save_metrics(results[0], thresholds[0], out_path='outputs/metrics_teotihuacan_0.002.json')
   pipeline.run_and_save_geojson(thresholds[0], out_path='outputs/teotihuacan_0.002.geojson')
   ```

---

## API Reference

### `OwlViTPipeline`

* **Constructor**: `OwlViTPipeline(experiment_name: str)`

  * `experiment_name`: Identifier for outputs and logs

* **Methods**:

  * `load_image(path: str, image_name: str)` – Load a geospatial image for inference.
  * `load_prompts(file_path: str)` – Load newline-delimited prompts.
  * `run_threshold_experiments(thresholds: List[float]) -> List[Dict]` – Perform inference at each threshold.
  * `save_visualisation(result: Dict, threshold: float, out_path: str) -> None` – Save annotated image.
  * `save_metrics(result: Dict, threshold: float, out_path: str) -> None` – Save JSON metrics.
  * `run_and_save_geojson(threshold: float, out_path: str) -> None` – Export detections to GeoJSON.

---

## Running Benchmarks

The `run_benchmark.py` script executes all notebooks under `experiments/` using Papermill, collects JSON metrics, and writes `benchmark_results.csv`:

```bash
python run_benchmark.py --data-dir data/images --prompts prompts.txt --output-dir outputs/
```

Generates:

* `outputs/benchmark_results.csv`
* Visualisations & GeoJSON for each experiment

---

## Contributing

1. Fork the repo
2. Create a branch (`git checkout -b feature/my-feature`)
3. Commit changes (`git commit -m 'Add feature'`)
4. Push and open a Pull Request

Ensure tests cover new code and follow PEP 8.

---

## License

Licensed under the [MIT License](LICENSE).
