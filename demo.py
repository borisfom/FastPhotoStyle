"""
Copyright (C) 2018 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

from __future__ import print_function
import argparse

import torch

import process_stylization
from photo_wct import PhotoWCT
from photo_gif import GIFSmoothing
parser = argparse.ArgumentParser(description='Photorealistic Image Stylization')
parser.add_argument('--model', default='./PhotoWCTModels/photo_wct.pth',
                    help='Path to the PhotoWCT model. These are provided by the PhotoWCT submodule, please use `git submodule update --init --recursive` to pull.')
parser.add_argument('--content_image_path', default='./images/content1.png')
parser.add_argument('--content_seg_path', default=[])
parser.add_argument('--style_image_path', default='./images/style1.png')
parser.add_argument('--style_seg_path', default=[])
parser.add_argument('--output_image_path', default='./results/example1.png')
parser.add_argument('--save_intermediate', action='store_true', default=False)
parser.add_argument('--fast', action='store_true', default=False)
parser.add_argument('--no_post', action='store_true', default=False)
parser.add_argument('--cuda', type=int, default=1, help='Enable CUDA.')
parser.add_argument("--export_onnx", type=str, help="export ONNX model to a given file")
parser.add_argument("--engine", type=str, help="run serialized TRT engine")
parser.add_argument("--onnx", type=str, help="run ONNX model via TRT")
parser.add_argument('--verbose', action='store_true', default = False, help='toggles verbose')
parser.add_argument("-d", "--data_type", default=32, type=int, choices=[8, 16, 32], help="Supported data type i.e. 8, 16, 32 bit")
    
    
args = parser.parse_args()

# Load model
p_wct = PhotoWCT(args)

if not p_wct.load():
    exit(1)

p_pro = GIFSmoothing(r=35, eps=0.001)

if args.cuda:
    p_wct.cuda(0)
    
process_stylization.stylization(
    stylization_module=p_wct,
    smoothing_module=p_pro,
    content_image_path=args.content_image_path,
    style_image_path=args.style_image_path,
    content_seg_path=args.content_seg_path,
    style_seg_path=args.style_seg_path,
    output_image_path=args.output_image_path,
    cuda=args.cuda,
    save_intermediate=args.save_intermediate,
    no_post=args.no_post,
    args=args
)
