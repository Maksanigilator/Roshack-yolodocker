"""
Download all duck datasets from Roboflow and merge with HuggingFace dataset.
Usage: python3 datasets/download_all.py --key YOUR_ROBOFLOW_API_KEY
"""

import argparse
import os
import shutil
import glob
import random
import yaml
from roboflow import Roboflow

DATASETS_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_DIR = os.path.join(DATASETS_DIR, "raw")
MERGED_DIR = os.path.join(DATASETS_DIR, "ducks-merged")

ROBOFLOW_DATASETS = [
    {
        "workspace": "duckies-1cfdk",
        "project": "real-world-duckietown-duckies",
        "version": 1,
        "keep_classes": ["Duckies"],
        "description": "Real World DuckieTown Duckies (2001 images)",
    },
    {
        "workspace": "ducks-pm7jq",
        "project": "aerial-only-duck-view",
        "version": 1,
        "keep_classes": "__all_except_null__",
        "description": "Aerial Only Duck View",
    },
    {
        "workspace": "patos",
        "project": "rubber-ducks",
        "version": 1,
        "keep_classes": ["rubber_duck"],
        "description": "Rubber Ducks by Patos",
    },
    {
        "workspace": "duckies",
        "project": "duckietown-hosv3",
        "version": 1,
        "keep_classes": ["Duckie"],
        "description": "DuckieTown HOSv3 (Duckie only)",
    },
]


def download_roboflow_datasets(api_key):
    rf = Roboflow(api_key=api_key)
    os.makedirs(RAW_DIR, exist_ok=True)

    for ds in ROBOFLOW_DATASETS:
        dst = os.path.join(RAW_DIR, ds["project"])
        if os.path.exists(dst):
            print(f"[skip] {ds['project']} already downloaded")
            continue
        print(f"[download] {ds['project']}...")
        project = rf.workspace(ds["workspace"]).project(ds["project"])
        dataset = project.version(ds["version"]).download("yolov8", location=dst)
        print(f"[done] {ds['project']}")


def parse_yaml_classes(yaml_path):
    with open(yaml_path) as f:
        data = yaml.safe_load(f)
    names = data.get("names", {})
    if isinstance(names, list):
        return {i: name for i, name in enumerate(names)}
    return {int(k): v for k, v in names.items()}


def remap_labels(label_path, class_map, keep_classes, new_class_id=0):
    """Read label file, keep only matching classes, remap to new_class_id."""
    if not os.path.exists(label_path):
        return []
    lines = []
    with open(label_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            cls_id = int(parts[0])
            cls_name = class_map.get(cls_id, "")

            if keep_classes == "__all_except_null__":
                if cls_name.lower() in ("null", "none", ""):
                    continue
            else:
                if cls_name not in keep_classes:
                    continue

            parts[0] = str(new_class_id)
            lines.append(" ".join(parts))
    return lines


def merge_datasets():
    os.makedirs(MERGED_DIR, exist_ok=True)
    img_counter = 0
    split_counts = {"train": 0, "val": 0, "test": 0}

    for split in ["train", "valid", "val", "test"]:
        target_split = "val" if split == "valid" else split
        img_dir = os.path.join(MERGED_DIR, target_split, "images")
        lbl_dir = os.path.join(MERGED_DIR, target_split, "labels")
        os.makedirs(img_dir, exist_ok=True)
        os.makedirs(lbl_dir, exist_ok=True)

    for ds in ROBOFLOW_DATASETS:
        src = os.path.join(RAW_DIR, ds["project"])
        if not os.path.exists(src):
            print(f"[warn] {ds['project']} not found, skipping")
            continue

        yaml_files = glob.glob(os.path.join(src, "*.yaml")) + glob.glob(os.path.join(src, "data.yaml"))
        if not yaml_files:
            print(f"[warn] no yaml found in {ds['project']}, skipping")
            continue

        class_map = parse_yaml_classes(yaml_files[0])
        print(f"[merge] {ds['project']}: classes={class_map}, keep={ds['keep_classes']}")

        for split in ["train", "valid", "val", "test"]:
            target_split = "val" if split == "valid" else split
            img_src = os.path.join(src, split, "images")
            lbl_src = os.path.join(src, split, "labels")

            if not os.path.isdir(img_src):
                continue

            for img_file in sorted(os.listdir(img_src)):
                if not img_file.lower().endswith((".jpg", ".jpeg", ".png")):
                    continue

                base = os.path.splitext(img_file)[0]
                ext = os.path.splitext(img_file)[1]
                lbl_file = base + ".txt"

                new_lines = remap_labels(
                    os.path.join(lbl_src, lbl_file),
                    class_map,
                    ds["keep_classes"],
                )

                if not new_lines:
                    continue

                new_name = f"duck_{img_counter:06d}"
                img_counter += 1

                shutil.copy2(
                    os.path.join(img_src, img_file),
                    os.path.join(MERGED_DIR, target_split, "images", new_name + ext),
                )
                with open(os.path.join(MERGED_DIR, target_split, "labels", new_name + ".txt"), "w") as f:
                    f.write("\n".join(new_lines))

                split_counts[target_split] = split_counts.get(target_split, 0) + 1

    hf_src = os.path.join(DATASETS_DIR, "rubber-ducks-yolo")
    if os.path.exists(hf_src):
        print("[merge] HuggingFace rubber-ducks-yolo")
        for split in ["train", "val"]:
            img_src = os.path.join(hf_src, split, "images")
            lbl_src = os.path.join(hf_src, split, "labels")
            if not os.path.isdir(img_src):
                continue
            for img_file in sorted(os.listdir(img_src)):
                if not img_file.lower().endswith((".jpg", ".jpeg", ".png")):
                    continue
                base = os.path.splitext(img_file)[0]
                ext = os.path.splitext(img_file)[1]
                lbl_path = os.path.join(lbl_src, base + ".txt")
                if not os.path.exists(lbl_path):
                    continue
                new_name = f"duck_{img_counter:06d}"
                img_counter += 1
                shutil.copy2(
                    os.path.join(img_src, img_file),
                    os.path.join(MERGED_DIR, split, "images", new_name + ext),
                )
                shutil.copy2(lbl_path, os.path.join(MERGED_DIR, split, "labels", new_name + ".txt"))
                split_counts[split] = split_counts.get(split, 0) + 1

    dataset_yaml = {
        "path": "/root/ros_ws/datasets/ducks-merged",
        "train": "train/images",
        "val": "val/images",
        "names": {0: "duck"},
    }
    with open(os.path.join(MERGED_DIR, "dataset.yaml"), "w") as f:
        yaml.dump(dataset_yaml, f, default_flow_style=False)

    print(f"\nMerged dataset: {MERGED_DIR}")
    for k, v in split_counts.items():
        print(f"  {k}: {v} images")
    print(f"  total: {sum(split_counts.values())} images")
    print(f"  class: 0 = duck")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--key", required=True, help="Roboflow API key")
    args = parser.parse_args()

    download_roboflow_datasets(args.key)
    merge_datasets()
