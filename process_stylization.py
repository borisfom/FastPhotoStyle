"""
Copyright (C) 2018 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

# 2018-03-30 Ming-Yu
#   - TODO: Better cuda interfacing code
#

from __future__ import print_function
import time
import numpy as np
from PIL import Image
from torch.autograd import Variable
import torchvision.transforms as transforms
import torchvision.utils as utils
import torch.nn as nn
from smooth_filter import smooth_filter


class Timer:
    def __init__(self, msg):
        self.msg = msg
        self.start_time = None
    
    def __enter__(self):
        self.start_time = time.time()
    
    def __exit__(self, exc_type, exc_value, exc_tb):
        print(self.msg % (time.time() - self.start_time))

def resize(img, max_small_len=840.0):
    ch = img.height
    cw = img.width
    cd = ch if ch < cw else cw
    cs = max_small_len / cd
    new_ch = int(cs * ch)
    new_cw = int(cs * cw)
    img.thumbnail((new_cw, new_ch), Image.BICUBIC)
    return new_cw, new_ch

def stylization(stylization_module, smoothing_module, content_image_path, style_image_path, content_seg_path, style_seg_path, output_image_path,
                cuda, save_intermediate, no_post):
    # Load image
    cont_img = Image.open(content_image_path).convert('RGB')
    styl_img = Image.open(style_image_path).convert('RGB')
    cont_pilimg = cont_img.copy()
    cw = cont_pilimg.width
    ch = cont_pilimg.height
    new_cw, new_ch = resize(cont_img)
    new_sw, new_sh = resize(styl_img)
    try:
        cont_seg = Image.open(content_seg_path)
        styl_seg = Image.open(style_seg_path)
        cont_seg.resize((new_cw,new_ch),Image.NEAREST)
        styl_seg.resize((new_sw,new_sh),Image.NEAREST)
    except:
        cont_seg = []
        styl_seg = []
    
    cont_img = transforms.ToTensor()(cont_img).unsqueeze(0)
    styl_img = transforms.ToTensor()(styl_img).unsqueeze(0)
    
    if cuda:
        cont_img = cont_img.cuda(0)
        styl_img = styl_img.cuda(0)
        stylization_module.cuda(0)
    
    cont_img = Variable(cont_img, volatile=True)
    styl_img = Variable(styl_img, volatile=True)
    
    cont_seg = np.asarray(cont_seg)
    styl_seg = np.asarray(styl_seg)

    if save_intermediate:
        with Timer("Elapsed time in stylization: %f"):
            stylized_img = stylization_module.transform(cont_img, styl_img, cont_seg, styl_seg)
        if ch != new_ch or cw != new_cw:
            stylized_img = nn.functional.upsample(stylized_img, size=(ch,cw), mode='bilinear')
        utils.save_image(stylized_img.data.cpu().float(), output_image_path, nrow=1, padding=0)

        with Timer("Elapsed time in propagation: %f"):
            out_img = smoothing_module.process(output_image_path, content_image_path)
        out_img.save(output_image_path)

        if not cuda:
            print("NotImplemented: The CPU version of smooth filter has not been implemented currently.")
            return

        if no_post is False:
            with Timer("Elapsed time in post processing: %f"):
                out_img = smooth_filter(output_image_path, content_image_path, f_radius=15, f_edge=1e-1)
        out_img.save(output_image_path)
    else:
        with Timer("Elapsed time in stylization: %f"):
            stylized_img = stylization_module.transform(cont_img, styl_img, cont_seg, styl_seg)
        if ch != new_ch or cw != new_cw:
            stylized_img = nn.functional.upsample(stylized_img, size=(ch,cw), mode='bilinear')
        grid = utils.make_grid(stylized_img.data, nrow=1, padding=0)
        ndarr = grid.mul(255).clamp(0, 255).byte().permute(1, 2, 0).cpu().numpy()
        out_img = Image.fromarray(ndarr)

        with Timer("Elapsed time in propagation: %f"):
            out_img = smoothing_module.process(out_img, cont_pilimg)

        if no_post is False:
            with Timer("Elapsed time in post processing: %f"):
                out_img = smooth_filter(out_img, cont_pilimg, f_radius=15, f_edge=1e-1)
        out_img.save(output_image_path)