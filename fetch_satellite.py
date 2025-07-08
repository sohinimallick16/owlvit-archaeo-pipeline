#!/usr/bin/env python3
"""
Download high-resolution satellite imagery for a bounding box without QGIS.
Uses ESRI World Imagery tiles (no API key required).

Usage:
    python fetch_satellite.py \
        --min-lat 19.692 --min-lon -98.860 \
        --max-lat 19.722 --max-lon -98.820 \
        --zoom 18 \
        --out teotihuacan.png
"""

import os
import math
import argparse
import requests
import io
from PIL import Image
import mercantile

ESRI_TILE_URL = (
    "https://server.arcgisonline.com/ArcGIS/rest/services/"
    "World_Imagery/MapServer/tile/{z}/{y}/{x}"
)

def parse_args():
    p = argparse.ArgumentParser(
        description="Fetch & stitch ESRI World Imagery tiles for a bbox"
    )
    p.add_argument("--min-lat", type=float, required=True, help="Minimum latitude")
    p.add_argument("--min-lon", type=float, required=True, help="Minimum longitude")
    p.add_argument("--max-lat", type=float, required=True, help="Maximum latitude")
    p.add_argument("--max-lon", type=float, required=True, help="Maximum longitude")
    p.add_argument(
        "--zoom",
        type=int,
        required=True,
        help="Tile zoom level (e.g. 16–20 for high-res)"
    )
    p.add_argument(
        "--out",
        type=str,
        default="out.png",
        help="Output PNG filename (will be saved into data/images/ if no path given)"
    )
    return p.parse_args()

def lonlat_to_pixel(lon: float, lat: float, zoom: int) -> (float, float):
    """
    Convert longitude/latitude to pixel coordinates at a given zoom level.
    """
    m = 256 * 2**zoom
    x = (lon + 180) / 360 * m
    y = (1 - (math.log(math.tan(math.radians(lat)) +
                       1 / math.cos(math.radians(lat))) / math.pi)) / 2 * m
    return x, y

def fetch_and_stitch(min_lat, min_lon, max_lat, max_lon, zoom, out_path):
    # Determine tile range
    tile_ul = mercantile.tile(min_lon, max_lat, zoom)  # upper-left
    tile_lr = mercantile.tile(max_lon, min_lat, zoom)  # lower-right

    x_min, x_max = min(tile_ul.x, tile_lr.x), max(tile_ul.x, tile_lr.x)
    y_min, y_max = min(tile_ul.y, tile_lr.y), max(tile_ul.y, tile_lr.y)
    cols = x_max - x_min + 1
    rows = y_max - y_min + 1

    print(f"Fetching {cols}×{rows} tiles at zoom {zoom}…")

    # Create blank canvas
    tile_size = 256
    canvas = Image.new("RGB", (cols * tile_size, rows * tile_size))

    # Download and paste each tile
    for ix, x in enumerate(range(x_min, x_max + 1)):
        for iy, y in enumerate(range(y_min, y_max + 1)):
            url = ESRI_TILE_URL.format(z=zoom, x=x, y=y)
            resp = requests.get(url, timeout=10)
            resp.raise_for_status()
            tile = Image.open(io.BytesIO(resp.content))
            canvas.paste(tile, (ix * tile_size, iy * tile_size))

    # Compute exact crop in pixel coordinates
    px_min, py_min = lonlat_to_pixel(min_lon, max_lat, zoom)
    px_max, py_max = lonlat_to_pixel(max_lon, min_lat, zoom)

    crop_x0 = int(px_min - x_min * tile_size)
    crop_y0 = int(py_min - y_min * tile_size)
    crop_x1 = int(px_max - x_min * tile_size)
    crop_y1 = int(py_max - y_min * tile_size)

    cropped = canvas.crop((crop_x0, crop_y0, crop_x1, crop_y1))

    # Ensure saving into data/images/ if no directory specified
    if not os.path.dirname(out_path):
        images_dir = os.path.join(os.getcwd(), "data", "images")
        os.makedirs(images_dir, exist_ok=True)
        out_path = os.path.join(images_dir, out_path)

    cropped.save(out_path)
    print(f"Saved stitched image to {out_path}")

if __name__ == "__main__":
    args = parse_args()
    fetch_and_stitch(
        args.min_lat, args.min_lon,
        args.max_lat, args.max_lon,
        args.zoom, args.out
    )
