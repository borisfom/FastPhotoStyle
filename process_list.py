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
parser.add_argument('--no_post', action='store_true', default=False)
parser.add_argument('--outp_img_folder', type=str, default='examples_stan/results')
parser.add_argument('--cont_img_folder', type=str, default='examples_stan/content_img')
parser.add_argument('--cont_seg_folder', type=str, default='examples_stan/content_seg')
parser.add_argument('--cont_list', type=str, default='examples_stan/list_content.txt')
parser.add_argument('--styl_img_folder', type=str, default='examples_stan/style_img')
parser.add_argument('--styl_seg_folder', type=str, default='examples_stan/style_seg')
parser.add_argument('--styl_list', type=str, default='examples_stan/list_style.txt')
parser.add_argument('--cont_img_ext', type=str, default='.png')
parser.add_argument('--cont_seg_ext', type=str, default='.pgm')
parser.add_argument('--styl_img_ext', type=str, default='.png')
parser.add_argument('--styl_seg_ext', type=str, default='.pgm')
parser.add_argument('--cont_seg_mapping', type=str, default='')
parser.add_argument('--styl_seg_mapping', type=str, default='')
parser.add_argument('--num_styles', type=int, default=10)
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
    p_pro = GIFSmoothing(r=35, eps=0.001)
else:
    p_pro = Propagator()

if args.cont_seg_mapping != '':
    import pickle
    cont_remapping = process_stylization.ReMapping()
    with open(args.cont_seg_mapping, 'rb') as f:
        cont_remapping.remapping = pickle.load(f)
else:
    cont_remapping = None

if args.styl_seg_mapping != '':
    import pickle
    styl_remapping = process_stylization.ReMapping()
    with open(args.styl_seg_mapping, 'rb') as f:
        styl_remapping.remapping = pickle.load(f)
else:
    styl_remapping = None

for f in content_list:
    print("Process " + f)
    f = f.strip()
    random.shuffle(style_list)
    s = 0
    # for s in range(0,args.num_styles):
    while s < args.num_styles:
        style_f = style_list[s].strip()
        content_image_path = os.path.join(args.cont_img_folder, f)
        content_seg_path = os.path.join(args.cont_seg_folder, f).replace(args.cont_img_ext, args.cont_seg_ext)
        style_image_path = os.path.join(args.styl_img_folder, style_f)
        style_seg_path = os.path.join(args.styl_seg_folder, style_f).replace(args.styl_img_ext, args.styl_seg_ext)
        output_image_path = os.path.join(args.outp_img_folder+"%02d"%s, f)
        directory = os.path.dirname(output_image_path)
        if not os.path.exists(directory):
            os.makedirs(directory)
        try:
            process_stylization.stylization(
                stylization_module=p_wct,
                smoothing_module=p_pro,
                content_image_path=content_image_path,
                style_image_path=style_image_path,
                content_seg_path=content_seg_path,
                style_seg_path=style_seg_path,
                output_image_path=output_image_path,
                cuda=args.cuda,
                save_intermediate=args.save_intermediate,
                no_post=args.no_post,
                cont_seg_remapping=cont_remapping,
                styl_seg_remapping=styl_remapping
            )
            s += 1
        except:
            continue
