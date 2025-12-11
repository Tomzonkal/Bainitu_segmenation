import json
import cv2
import numpy as np
import os
from shapely.geometry import Polygon
from datasets.datasets import Dataset


class GridSegmentsCreator:
    """
    Splits images into an M×N grid.
    Uses LabelMe-style metadata to assign a majority phase label to each tile.
    """

    def __init__(self, input_dataset: Dataset):
        self.input_dataset = input_dataset
        self.grid_rows = None
        self.grid_cols = None
        self.output_dataset = None

        # Stats
        self.total_tiles = 0
        self.mixed_tiles = 0
        self.bainite_tiles = 0
        self.martensite_tiles = 0
        self.unlabeled_tiles = 0

    # -------------------------------------------------------------------------
    # Loaders
    # -------------------------------------------------------------------------

    def _load_image(self, image_path):
        img = cv2.imread(image_path)
        if img is None:
            raise FileNotFoundError(f"Could not load image: {image_path}")
        return img

    def _load_labelme_json(self, json_path):
        """
        Loads LabelMe JSON and returns fixed shapely polygons.
        Invalid polygons are auto-corrected using .buffer(0).
        """
        with open(json_path, "r") as f:
            data = json.load(f)

        phase_polygons = {"martensite": [], "bainite": []}

        for shape in data.get("shapes", []):
            label = shape["label"].lower()
            pts = shape.get("points", [])

            if len(pts) < 3:
                continue  # not a polygon

            poly = Polygon(pts)

            # Fix invalid geometry
            try:
                poly = poly.buffer(0)
            except Exception:
                continue

            if poly.is_empty:
                continue
            if not poly.is_valid:
                continue

            if label in phase_polygons:
                phase_polygons[label].append(poly)

        return phase_polygons

    # -------------------------------------------------------------------------
    # Labeling logic
    # -------------------------------------------------------------------------

    def _tile_label(self, tile_poly, phase_polygons):
        """
        Computes intersection areas between a grid tile and phase polygons.
        - Returns dominant label or "mixed" or "unlabeled".
        """

        areas = {}
        for phase, polys in phase_polygons.items():
            area_sum = 0.0
            for p in polys:
                inter = tile_poly.intersection(p)
                if not inter.is_empty:
                    area_sum += inter.area
            areas[phase] = area_sum

        # No intersection at all?
        if areas["martensite"] == 0 and areas["bainite"] == 0:
            return "unlabeled"

        # Majority decision
        if areas["martensite"] > areas["bainite"]:
            if areas["martensite"] >= 0.5 * (areas["martensite"] + areas["bainite"]):
                return "martensite"
        if areas["bainite"] > areas["martensite"]:
            if areas["bainite"] >= 0.5 * (areas["martensite"] + areas["bainite"]):
                return "bainite"

        return "mixed"

    # -------------------------------------------------------------------------
    # Saving functionality
    # -------------------------------------------------------------------------

    def _save_segment_image(self, base_name, tile_id, tile_img, output_dataset):

        outdir = os.path.join(output_dataset.image_data_path,
                              output_dataset.dataset_name)
        os.makedirs(outdir, exist_ok=True)

        filename = f"{base_name}_tile_{tile_id}.png"
        cv2.imwrite(os.path.join(outdir, filename), tile_img)

    def _save_segment_metadata(self, base_name, tile_id, label, tile_shape, output_dataset):

        outdir = os.path.join(output_dataset.image_label_data_path,
                              output_dataset.dataset_name)
        os.makedirs(outdir, exist_ok=True)

        metadata = {
            "base_name": base_name,
            "tile_id": tile_id,
            "label": label,
            "tile_shape": tile_shape
        }

        filename = f"{base_name}_tile_{tile_id}.json"
        with open(os.path.join(outdir, filename), "w") as f:
            json.dump(metadata, f, indent=4)

    # -------------------------------------------------------------------------
    # Main method
    # -------------------------------------------------------------------------

    def create_segments(self, grid_parameters, segmentation_out_dataset):
        metadata = self.input_dataset.load_meta_data()
        self.grid_rows = grid_parameters["grid_rows"]
        self.grid_cols = grid_parameters["grid_cols"]

        for _, row in metadata.iterrows():
            image_path = row["image_path"]
            json_path = row["json_path"]

            if json_path is None:
                print(f"Warning: No JSON path for {image_path}, skipping...")
                continue

            base_name = os.path.splitext(os.path.basename(json_path))[0]

            image = self._load_image(image_path)
            h, w = image.shape[:2]

            phase_polygons = self._load_labelme_json(json_path)

            tile_h = h // self.grid_rows
            tile_w = w // self.grid_cols

            tile_id = 0

            for r in range(self.grid_rows):
                for c in range(self.grid_cols):

                    x1 = c * tile_w
                    y1 = r * tile_h
                    x2 = x1 + tile_w
                    y2 = y1 + tile_h

                    tile_img = image[y1:y2, x1:x2]

                    tile_poly = Polygon([
                        (x1, y1), (x2, y1),
                        (x2, y2), (x1, y2)
                    ])

                    label = self._tile_label(tile_poly, phase_polygons)

                    # Save output
                    self._save_segment_image(base_name, tile_id, tile_img, segmentation_out_dataset)
                    self._save_segment_metadata(base_name, tile_id, label, tile_img.shape, segmentation_out_dataset)

                    # Stats
                    self.total_tiles += 1
                    if label == "martensite":
                        self.martensite_tiles += 1
                    elif label == "bainite":
                        self.bainite_tiles += 1
                    elif label == "mixed":
                        self.mixed_tiles += 1
                    elif label == "unlabeled":
                        self.unlabeled_tiles += 1

                    tile_id += 1

    # -------------------------------------------------------------------------
    # Statistics
    # -------------------------------------------------------------------------

    def get_statistics(self):
        return {
            "total_tiles": self.total_tiles,
            "bainite_tiles": self.bainite_tiles,
            "martensite_tiles": self.martensite_tiles,
            "mixed_tiles": self.mixed_tiles,
            "unlabeled_tiles": self.unlabeled_tiles,
        }
