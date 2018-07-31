"""
Copyright (C) 2018 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

from __future__ import print_function
import argparse
import os
import torch
import sys
import onnx
import cv2
import numpy as np
from torch.autograd import Variable

sys.path.append('./stylization')
from scipy.misc import imread, imresize
from torchvision import transforms
from lib.nn import user_scattered_collate, async_copy_to
from segmentation.dataset import round2nearest_multiple
from segmentation.models import ModelBuilder, SegmentationModule

from lib.utils import as_numpy, mark_volatile
from photo_wct import PhotoWCT
from photo_gif import GIFSmoothing
import process_stylization_ade20k

# Global variables
BASE_BLACK_IMAGE_NAME = 'tmp_base_black_img.png'
CONTENT_IMAGE_NAME ='tmp_content_img.png'
STYLE_IMAGE_NAME = 'tmp_style_img.png'
CONTENT_SEG_NAME ='tmp_content_seg.pgm'
STYLE_SEG_NAME = 'tmp_style_seg.pgm'
VIS_CONTENT_SEG_NAME ='tmp_vis_content_seg.pgm'
VIS_STYLE_SEG_NAME = 'tmp_vis_style_seg.pgm'
SEG_STYLIZATION_OUTPUT_NAME = 'tmp_seg_output.png'
STYLIZATION_OUTPUT_NAME = 'tmp_output.png'
BUFFER_IMAGE_NAME = 'tmp.png'
BASE_WIDTH = 768
SMALL_BASE_WIDTH = 712
BORDER = 4
IMAGE_WIDTH = BASE_WIDTH*3
IMAGE_HEIGHT = BASE_WIDTH*1

parser = argparse.ArgumentParser(description='Photorealistic Image Stylization')

# Below are from Segmentation Network
parser.add_argument('--model_path', help='folder to model path', default='baseline-resnet50_dilated8-ppm_bilinear_deepsup')
parser.add_argument('--suffix', default='_epoch_20.pth', help="which snapshot to load")
parser.add_argument('--arch_encoder', default='resnet50_dilated8', help="architecture of net_encoder")
parser.add_argument('--arch_decoder', default='ppm_bilinear_deepsup', help="architecture of net_decoder")
parser.add_argument('--fc_dim', default=2048, type=int, help='number of features between encoder and decoder')
parser.add_argument('--num_val', default=-1, type=int, help='number of images to evalutate')
parser.add_argument('--num_class', default=150, type=int, help='number of classes')
parser.add_argument('--batch_size', default=1, type=int, help='batchsize. current only supports 1')
parser.add_argument('--imgSize', default=[300, 400, 500, 600], nargs='+', type=int, help='list of input image sizes.' 'for multiscale testing, e.g. 300 400 500')
parser.add_argument('--imgMaxSize', default=1000, type=int, help='maximum input image size of long edge')
parser.add_argument('--padding_constant', default=8, type=int, help='maxmimum downsampling rate of the network')
parser.add_argument('--segm_downsampling_rate', default=8, type=int, help='downsampling rate of the segmentation label')
parser.add_argument('--gpu_id', default=0, type=int, help='gpu_id for evaluation')

# Below are from FastPhotStyle
parser.add_argument('--folder', type=str, default='examples')
parser.add_argument('--beta', type=float, default=0.9999)
parser.add_argument('--cont_img_ext', type=str, default='.png')
parser.add_argument('--cont_seg_ext', type=str, default='.pgm')
parser.add_argument('--styl_img_ext', type=str, default='.png')
parser.add_argument('--styl_seg_ext', type=str, default='.pgm')
parser.add_argument('--label_mapping', type=str, default='segmentation/semantic_rel.npy')

parser.add_argument('--model', default='./PhotoWCTModels/photo_wct.pth',
                    help='Path to the PhotoWCT model. These are provided by the PhotoWCT submodule, please use `git submodule update --init --recursive` to pull.')
parser.add_argument('--content_image_path', default='./content_img/in1.png')
parser.add_argument('--content_seg_path', default='./content.pgm')
parser.add_argument('--style_image_path', default='./style_img/in1.png')
parser.add_argument('--style_seg_path', default='./style.pgm')
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


def adjust_image_size(cont_img):
        MINSIZE = 240
        MAXSIZE = 1920
        ow = cont_img.shape[1]
        oh = cont_img.shape[0]
        if max(ow, oh) <= MINSIZE:
            if ow > oh:
                new_img = cv2.resize(cont_img, dsize=(int(ow * 1.0 / oh * MINSIZE), MINSIZE))
            else:
                new_img = cv2.resize(cont_img, dsize=(MINSIZE, int(oh * 1.0 / ow * MINSIZE)))
        elif min(ow, oh) >= MAXSIZE:
            if ow > oh:
                new_img = cv2.resize(cont_img, dsize=(MAXSIZE, int(oh * 1.0 / ow * MAXSIZE)))
            else:
                new_img = cv2.resize(cont_img, dsize=(int(ow * 1.0 / oh * MAXSIZE), MAXSIZE))
        else:
            new_img =  cont_img
        nw = new_img.shape[1]
        nh = new_img.shape[0]
        print("Resize image: (%d,%d)->(%d,%d)" % (ow, oh, nw, nh))
        return new_img

    
def segment_this_img(f):
    img = imread(f, mode='RGB')
    img = img[:, :, ::-1]  # BGR to RGB!!!
    ori_height, ori_width, _ = img.shape
    img_resized_list = []
    for this_short_size in args.imgSize:
        scale = this_short_size / float(min(ori_height, ori_width))
        target_height, target_width = int(ori_height * scale), int(ori_width * scale)
        target_height = round2nearest_multiple(target_height, args.padding_constant)
        target_width = round2nearest_multiple(target_width, args.padding_constant)
        img_resized = cv2.resize(img.copy(), (target_width, target_height))
        img_resized = img_resized.astype(np.float32)
        img_resized = img_resized.transpose((2, 0, 1))
        img_resized = transform(torch.from_numpy(img_resized))
        img_resized = torch.unsqueeze(img_resized, 0)
        img_resized_list.append(img_resized)
    input = dict()
    input['img_ori'] = img.copy()
    input['img_data'] = [x.contiguous() for x in img_resized_list]
    segSize = (img.shape[0],img.shape[1])
    print(segSize)
    with torch.no_grad():
        pred = torch.zeros(1, args.num_class, segSize[0], segSize[1])
        for timg in img_resized_list:
            # forward pass
            pred_tmp = segmentation_module(timg.cuda())
            print (pred_tmp.shape)
            pred_tmp = pred_tmp.cpu()/ len(args.imgSize)
            print (pred_tmp.shape)
            
            pred = pred + pred_tmp
            
        _, preds = torch.max(pred, dim=1)
        preds = as_numpy(preds.squeeze(0))
    return preds
    
args = parser.parse_args()

# Load model
p_wct = PhotoWCT(args)

p_pro = GIFSmoothing(r=35, eps=0.01)
segReMapping = process_stylization_ade20k.SegReMapping(args.label_mapping)

# absolute paths of model weights
SEG_NET_PATH = 'segmentation'
args.weights_encoder = os.path.join(SEG_NET_PATH,args.model_path, 'encoder' + args.suffix)
args.weights_decoder = os.path.join(SEG_NET_PATH,args.model_path, 'decoder' + args.suffix)
args.arch_encoder = 'resnet50'
# args.arch_decoder = 'ppm_bilinear_deepsup'
args.fc_dim = 2048
# Network Builders
builder = ModelBuilder()
net_encoder = builder.build_encoder(arch=args.arch_encoder, fc_dim=args.fc_dim, weights=args.weights_encoder)
net_decoder = builder.build_decoder(arch=args.arch_decoder, fc_dim=args.fc_dim, num_class=args.num_class, weights=args.weights_decoder, use_softmax=True)
crit = torch.nn.NLLLoss(ignore_index=-1)
segmentation_module = SegmentationModule(net_encoder, net_decoder, crit, args)
segmentation_module.cuda()
segmentation_module.eval()
transform = transforms.Compose([transforms.Normalize(mean=[102.9801, 115.9465, 122.7717], std=[1., 1., 1.])])

style_img = cv2.imread(args.style_image_path)
new_style_img = adjust_image_size(style_img)
if args.export_onnx:
    assert args.export_onnx.endswith(".onnx"), "Export model file should end with .onnx"
    torch.onnx._export(segmentation_module, Variable(transforms.ToTensor()(new_style_img).unsqueeze(0).cuda(0), requires_grad=False), f='segm-'+args.export_onnx, verbose=args.verbose)
cv2.imwrite(STYLE_IMAGE_NAME, new_style_img)
style_seg = segment_this_img(STYLE_IMAGE_NAME)
cv2.imwrite(STYLE_SEG_NAME,style_seg)


cont_img = cv2.imread(args.content_image_path)
new_cont_img = adjust_image_size(cont_img)
cv2.imwrite(CONTENT_IMAGE_NAME, new_cont_img)    
cont_seg = segment_this_img(CONTENT_IMAGE_NAME)
cv2.imwrite(CONTENT_SEG_NAME,cont_seg)

if not p_wct.load():
    exit(1)

p_pro = GIFSmoothing(r=35, eps=0.001)

if args.cuda:
    p_wct.cuda(0)
        
if args.export_onnx:
    assert args.export_onnx.endswith(".onnx"), "Export model file should end with .onnx"

    torch.onnx._export(stylization_module, [new_cont_img, new_style_img, cont_seg, style_seg],
                       f=args.export_onnx, verbose=args.verbose)
    exit(0)
    
process_stylization_ade20k.stylization(
    stylization_module=p_wct,
    smoothing_module=p_pro,
    content_image_path=CONTENT_IMAGE_NAME,
    style_image_path=STYLE_IMAGE_NAME,
    content_seg_path=CONTENT_SEG_NAME,
    style_seg_path=STYLE_SEG_NAME,
    output_image_path=args.output_image_path,
    cuda=args.cuda,
    save_intermediate=args.save_intermediate,
    no_post=args.no_post,
)
