# -*- coding: utf-8 -*-
"""
Created on Thu Sep 18 11:05:22 2025

@author: Acer
"""

"""
Convert COCO-Text V2.0 JSON to PaddleOCR detection format.

Input:
  --coco_json path/to/cocotext.json
  --images_dir path/to/images
  --out_dir output_dir (will contain train_list.txt and images copied or symlinked)

Output format (PaddleOCR detection):
  Each line in train_list.txt:
    /abs/path/to/image.jpg x1,y1,x2,y2,x3,y3,x4,y4,,,transcription
  (polygons as comma separated ints, then a tab or space, then transcription)
"""

import os
import argparse
import json
from pathlib import Path
from shutil import copyfile

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--coco_json", required=True)
    p.add_argument("--images_dir", required=True)
    p.add_argument("--out_dir", required=True)
    return p.parse_args()

def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    img_out = out_dir / "images"
    img_out.mkdir(exist_ok=True)

    with open(args.coco_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    id_to_filename = {img["id"]: img["file_name"] for img in data["images"]}
    anns_by_image = {}
    for ann in data["annotations"]:
        img_id = ann["image_id"]
        anns_by_image.setdefault(img_id, []).append(ann)

    train_list_path = out_dir / "train_list.txt"
    lines = []
    for img_id, anns in anns_by_image.items():
        fname = id_to_filename[img_id]
        src = Path(args.images_dir) / fname
        if not src.exists():
            print(f"WARNING: image not found {src}")
            continue
        dst = img_out / fname
        try:
            copyfile(str(src), str(dst))
        except Exception as e:
            print("copy error", e)
            continue

        # For each annotation, compute polygon and transcription
        # COCO-Text often stores bbox and mask; prefer polygon if available
        poly_strings = []
        for ann in anns:
            # ann may have 'mask' or 'segmentation' or bbox
            seg = ann.get("segmentation", [])
            if isinstance(seg, list) and len(seg) > 0:
                # flatten first segmentation
                poly = seg[0]
                # ensure length even
                pts = []
                for x in poly:
                    pts.append(str(int(round(x))))
                poly_str = ",".join(pts)
            else:
                # fallback to bbox -> make rectangle polygon
                x,y,w,h = ann.get("bbox", [0,0,0,0])
                pts = [x,y,x+w,y,x+w,y+h,x,y+h]
                pts = [str(int(round(x))) for x in pts]
                poly_str = ",".join(pts)
            txt = ann.get("utf8_string") or ann.get("text") or ann.get("transcription") or ann.get("txt", "")
            # escape newline and tabs
            txt = str(txt).replace("\n", " ").replace("\t", " ")
            # For PaddleOCR det format: polygon transcription pairs separated by space
            # We'll join multiple instances with '|'
            poly_strings.append(f"{poly_str},{txt}")
        # join instances by space (PaddleOCR can parse multiple)
        line = f"{str(dst)} " + " ".join(poly_strings)
        lines.append(line)

    with open(train_list_path, "w", encoding="utf-8") as f:
        for ln in lines:
            f.write(ln + "\n")
    print("Wrote", train_list_path)

if __name__ == "__main__":
    main()
