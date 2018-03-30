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
from smooth_filter import smooth_filter


class Timer:
    def __init__(self, msg):
        self.msg = msg
        self.start_time = None
    
    def __enter__(self):
        self.start_time = time.time()
    
    def __exit__(self, exc_type, exc_value, exc_tb):
        print(self.msg % (time.time() - self.start_time))


def stylization(stylization_module, smoothing_module, content_image_path, style_image_path, content_seg_path, style_seg_path, output_image_path,
                cuda, save_intermediate):
    # Load image
    cont_img = Image.open(content_image_path).convert('RGB')
    styl_img = Image.open(style_image_path).convert('RGB')
    cont_pilimg = cont_img.copy()
    try:
        cont_seg = Image.open(content_seg_path)
        styl_seg = Image.open(style_seg_path)
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
        utils.save_image(stylized_img.data.cpu().float(), output_image_path, nrow=1, padding=0)

        with Timer("Elapsed time in propagation: %f"):
            out_img = smoothing_module.process(output_image_path, content_image_path)
        out_img.save(output_image_path)

        if not cuda:
            print("NotImplemented: The CPU version of smooth filter has not been implemented currently.")
            return

        with Timer("Elapsed time in post processing: %f"):
            out_img = smooth_filter(output_image_path, content_image_path, f_radius=15, f_edge=1e-1)
        out_img.save(output_image_path)
    else:
        with Timer("Elapsed time in stylization: %f"):
            stylized_img = stylization_module.transform(cont_img, styl_img, cont_seg, styl_seg)
        out_img = transforms.ToPILImage()(stylized_img[0,::].data.cpu().float())

        with Timer("Elapsed time in propagation: %f"):
            out_img = smoothing_module.process(out_img, cont_pilimg)

        with Timer("Elapsed time in post processing: %f"):
            out_img = smooth_filter(out_img, cont_pilimg, f_radius=15, f_edge=1e-1)
        out_img.save(output_image_path)