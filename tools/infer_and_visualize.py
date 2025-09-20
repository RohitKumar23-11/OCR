# -*- coding: utf-8 -*-
"""
Created on Thu Sep 18 11:06:24 2025

@author: Acer
"""

"""
Simplified inference and visualization using PaddleOCR Python API for quick checks.
Requires `pip install paddleocr` or use PaddleOCR repo's APIs.

Usage example:
python tools/infer_and_visualize.py --image sample.jpg --save_dir out --lang en
"""
import argparse
from paddleocr import PaddleOCR, draw_ocr
from PIL import Image
import os

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--image", required=True)
    p.add_argument("--save_dir", default="out")
    p.add_argument("--lang", default="en")
    return p.parse_args()

def main():
    args = parse_args()
    ocr = PaddleOCR(use_angle_cls=True, lang=args.lang) # loads default models (may download)
    result = ocr.ocr(args.image, cls=True)
    img = Image.open(args.image).convert('RGB')
    boxes = [line[0] for line in result]
    txts = [line[1][0] for line in result]
    scores = [line[1][1] for line in result]
    im_show = draw_ocr(img, boxes, txts, scores, font_path=None)
    os.makedirs(args.save_dir, exist_ok=True)
    outfile = os.path.join(args.save_dir, os.path.basename(args.image))
    im_show.save(outfile)
    print("Saved visualization to", outfile)

if __name__ == "__main__":
    main()
