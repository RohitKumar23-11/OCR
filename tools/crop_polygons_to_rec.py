# -*- coding: utf-8 -*-
"""
Created on Thu Sep 18 11:06:02 2025

@author: Acer
"""

"""
Given detection polygons (PaddleOCR detection train_list.txt), crop each polygon to an image
and save as rec_images/<image_id>_<idx>.jpg and produce rec_train.txt with lines:
  rec_images/<image_id>_<idx>.jpg\ttranscription

This is useful to prepare recognition training data from detection polygon labels.
"""
import os, sys
import argparse
from pathlib import Path
import cv2
import numpy as np

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--det_list", required=True, help="PaddleOCR detection list (train_list.txt)")
    p.add_argument("--out_dir", required=True)
    return p.parse_args()

def crop_polygon(img, pts):
    # pts: Nx2 array
    rect = cv2.minAreaRect(np.array(pts, dtype=np.float32))
    box = cv2.boxPoints(rect)
    box = np.int0(box)
    width = int(rect[1][0])
    height = int(rect[1][1])
    src_pts = box.astype("float32")
    # target points
    dst_pts = np.array([[0, height-1], [0,0], [width-1,0], [width-1,height-1]], dtype="float32")
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    warped = cv2.warpPerspective(img, M, (width, height))
    return warped

def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    rec_img_dir = out_dir / "rec_images"
    rec_img_dir.mkdir(parents=True, exist_ok=True)
    rec_gt = out_dir / "rec_train.txt"
    lines_out = []
    with open(args.det_list, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(" ")
            img_path = parts[0]
            insts = parts[1:]
            img = cv2.imread(img_path)
            if img is None:
                print("Could not read", img_path)
                continue
            base = Path(img_path).stem
            for idx, inst in enumerate(insts):
                # inst format: x1,y1,x2,y2,...,text
                # find last comma before the text portion: text may contain commas -> assume transcription last token after last comma? Simpler: split by ',' then last token is text if text has spaces. But we expect last element to be the text
                tokens = inst.split(",")
                if len(tokens) < 9:
                    continue
                # polygon coords all numeric until the token where non-numeric starts -> but simpler: coords are pairs; transcription is last token element(s). We'll assume last token is transcription.
                # Let's take coords = tokens[:-1]; text = tokens[-1]
                coords = tokens[:-1]
                txt = tokens[-1]
                pts = []
                try:
                    for i in range(0, len(coords), 2):
                        x = float(coords[i]); y = float(coords[i+1])
                        pts.append([x,y])
                except Exception as e:
                    continue
                crop = crop_polygon(img, pts)
                out_name = f"{base}_{idx}.jpg"
                out_path = rec_img_dir / out_name
                cv2.imwrite(str(out_path), crop)
                lines_out.append(f"{str(out_path)}\t{txt}")
    with open(rec_gt, "w", encoding="utf-8") as f:
        for l in lines_out:
            f.write(l + "\n")
    print("Wrote recognition gt to", rec_gt)

if __name__ == "__main__":
    main()
