"""
Copyright (C) 2018 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

from __future__ import print_function
import argparse
import random
import os
import torch
from photo_wct import PhotoWCT
from photo_smooth import Propagator
from photo_gif import GIFSmoothing
import process_stylization

parser = argparse.ArgumentParser(description='Photorealistic Image Stylization')
parser.add_argument('--model', default='./PhotoWCTModels/photo_wct.pth',
                    help='Path to the PhotoWCT model. These are provided by the PhotoWCT submodule, please use `git submodule update --init --recursive` to pull.')
parser.add_argument('--cuda', type=bool, default=True, help='Enable CUDA.')
parser.add_argument('--save_intermediate', action='store_true', default=False)
parser.add_argument('--fast', action='store_true', default=False)
parser.add_argument('--outp_img_folder', type=str, default='examples_stan/results')
parser.add_argument('--cont_img_folder', type=str, default='examples_stan/content_img')
parser.add_argument('--cont_seg_folder', type=str, default='examples_stan/content_seg')
parser.add_argument('--cont_list', type=str, default='examples_stan/list_content.txt')
parser.add_argument('--styl_img_folder', type=str, default='examples_stan/style_img')
parser.add_argument('--styl_seg_folder', type=str, default='examples_stan/style_seg')
parser.add_argument('--styl_list', type=str, default='examples_stan/list_style.txt')
args = parser.parse_args()


with open(args.cont_list) as f:
    content_list = f.readlines()

with open(args.styl_list) as f:
    style_list = f.readlines()

# Load model
p_wct = PhotoWCT()
p_wct.load_state_dict(torch.load(args.model))
# Load Propagator
if args.fast==True:
    p_pro = GIFSmoothing(r=35, eps=0.3)
else:
    p_pro = Propagator()


for f in content_list:
    print("Process " + f)
    f = f.strip()
    random.shuffle(style_list)
    style_f = style_list[0].strip()
    content_image_path = os.path.join(args.cont_img_folder, f)
    content_seg_path = os.path.join(args.cont_seg_folder, f).replace(".png", ".pgm")
    style_image_path = os.path.join(args.styl_img_folder, style_f)
    style_seg_path = os.path.join(args.styl_seg_folder, style_f).replace(".png", ".pgm")
    output_image_path = os.path.join(args.outp_img_folder, f)
    directory = os.path.dirname(output_image_path)
    if not os.path.exists(directory):
        os.makedirs(directory)

    process_stylization.stylization(
        stylization_module=p_wct,
        smoothing_module=p_pro,
        content_image_path=content_image_path,
        style_image_path=style_image_path,
        content_seg_path=content_seg_path,
        style_seg_path=style_seg_path,
        output_image_path=output_image_path,
        cuda=args.cuda,
        save_intermediate=args.save_intermediate
    )
