#!/usr/bin/env python


#############################################################################
##
## Copyright (C) 2013 Riverbank Computing Limited.
## Copyright (C) 2010 Nokia Corporation and/or its subsidiary(-ies).
## All rights reserved.
##
## This file is part of the examples of PyQt.
##
## $QT_BEGIN_LICENSE:BSD$
## You may use this file under the terms of the BSD license as follows:
##
## "Redistribution and use in source and binary forms, with or without
## modification, are permitted provided that the following conditions are
## met:
##   * Redistributions of source code must retain the above copyright
##     notice, this list of conditions and the following disclaimer.
##   * Redistributions in binary form must reproduce the above copyright
##     notice, this list of conditions and the following disclaimer in
##     the documentation and/or other materials provided with the
##     distribution.
##   * Neither the name of Nokia Corporation and its Subsidiary(-ies) nor
##     the names of its contributors may be used to endorse or promote
##     products derived from this software without specific prior written
##     permission.
##
## THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
## "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
## LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
## A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
## OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
## SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
## LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
## DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
## THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
## (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
## OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE."
## $QT_END_LICENSE$
##
#############################################################################


from PyQt5.QtCore import QDir, Qt
from PyQt5.QtGui import QImage, QPainter, QPalette, QPixmap
from PyQt5.QtWidgets import (QAction, QApplication, QFileDialog, QLabel,
        QMainWindow, QMenu, QMessageBox, QScrollArea, QSizePolicy)
from PyQt5.QtPrintSupport import QPrintDialog, QPrinter
import sys
import os
import numpy as np
import argparse
import torch.nn as nn
import torch
import cv2
from scipy.misc import imread, imresize
from torchvision import transforms
from segmentation.dataset import round2nearest_multiple
from segmentation.models import ModelBuilder, SegmentationModule
from lib.nn import user_scattered_collate, async_copy_to
from lib.utils import as_numpy, mark_volatile
from photo_wct import PhotoWCT
from photo_gif import GIFSmoothing
import process_stylization_ade20k
import process_stylization

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

parser = argparse.ArgumentParser()
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
parser.add_argument('--model', default='./PhotoWCTModels/photo_wct.pth', help='Path to the PhotoWCT model. These are provided by the PhotoWCT submodule, please use `git submodule update --init --recursive` to pull.')
parser.add_argument('--save_intermediate', action='store_true', default=False)
parser.add_argument('--fast', action='store_true', default=False)
parser.add_argument('--no_post', action='store_true', default=False)
parser.add_argument('--folder', type=str, default='examples')
parser.add_argument('--beta', type=float, default=0.9999)
parser.add_argument('--cont_img_ext', type=str, default='.png')
parser.add_argument('--cont_seg_ext', type=str, default='.pgm')
parser.add_argument('--styl_img_ext', type=str, default='.png')
parser.add_argument('--styl_seg_ext', type=str, default='.pgm')
parser.add_argument('--label_mapping', type=str, default='segmentation/semantic_rel.npy')
args = parser.parse_args()


# Load model
p_wct = PhotoWCT()
p_wct.load_state_dict(torch.load(args.model))
p_pro = GIFSmoothing(r=35, eps=0.01)
segReMapping = process_stylization_ade20k.SegReMapping(args.label_mapping)

# absolute paths of model weights
SEG_NET_PATH = 'segmentation'
args.weights_encoder = os.path.join(SEG_NET_PATH,args.model_path, 'encoder' + args.suffix)
args.weights_decoder = os.path.join(SEG_NET_PATH,args.model_path, 'decoder' + args.suffix)
args.arch_encoder = 'resnet50_dilated8'
args.arch_decoder = 'ppm_bilinear_deepsup'
args.fc_dim = 2048
# Network Builders
builder = ModelBuilder()
net_encoder = builder.build_encoder(arch=args.arch_encoder, fc_dim=args.fc_dim, weights=args.weights_encoder)
net_decoder = builder.build_decoder(arch=args.arch_decoder, fc_dim=args.fc_dim, num_class=args.num_class, weights=args.weights_decoder, use_softmax=True)
crit = nn.NLLLoss(ignore_index=-1)
segmentation_module = SegmentationModule(net_encoder, net_decoder, crit)
segmentation_module.cuda()
segmentation_module.eval()
transform = transforms.Compose([transforms.Normalize(mean=[102.9801, 115.9465, 122.7717], std=[1., 1., 1.])])

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
    with torch.no_grad():
        pred = torch.zeros(1, args.num_class, segSize[0], segSize[1])
        for timg in img_resized_list:
            feed_dict = dict()
            feed_dict['img_data'] = timg.cuda()
            feed_dict = async_copy_to(feed_dict, args.gpu_id)
            # forward pass
            pred_tmp = segmentation_module(feed_dict, segSize=segSize)
            pred = pred + pred_tmp.cpu() / len(args.imgSize)
        _, preds = torch.max(pred, dim=1)
        preds = as_numpy(preds.squeeze(0))
    return preds


def overlay(img, pred_color, blend_factor=0.4):
    edges = cv2.Canny(pred_color,20,40)
    edges = cv2.dilate(edges, np.ones((5,5),np.uint8), iterations=1)
    out = (1-blend_factor)*img + blend_factor * pred_color
    edge_pixels = (edges==255)
    new_color = [0,0,255]
    for i in range(0,3):
        timg = out[:,:,i]
        timg[edge_pixels]=new_color[i]
        out[:,:,i] = timg
    return out


class ImageViewer(QMainWindow):
    def __init__(self):
        super(ImageViewer, self).__init__()
        self.buffer_image = np.zeros((IMAGE_HEIGHT,IMAGE_WIDTH), dtype=np.uint8)
        cv2.imwrite(BUFFER_IMAGE_NAME, self.buffer_image)
        self.base_black_image = np.zeros((BASE_WIDTH, BASE_WIDTH), dtype=np.uint8)
        cv2.imwrite(BASE_BLACK_IMAGE_NAME, self.base_black_image)
        self.content_printer = QPrinter()
        self.style_printer = QPrinter()
        self.scaleFactor = 0.0
        self.imageLabel = QLabel()
        self.imageLabel.setBackgroundRole(QPalette.Base)
        self.imageLabel.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        self.imageLabel.setScaledContents(False)
        self.scrollArea = QScrollArea()
        self.scrollArea.setBackgroundRole(QPalette.Dark)
        self.scrollArea.setWidget(self.imageLabel)
        self.setCentralWidget(self.scrollArea)
        self.createActions()
        self.createMenus()
        self.setWindowTitle("NVIDIA FastPhotoStyle Demo")
        self.resize(IMAGE_WIDTH, IMAGE_HEIGHT)

    def adjust_image_size(self, cont_img):
        MINSIZE = 240
        MAXSIZE = 960
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

    def open_content(self):
        fileName, _ = QFileDialog.getOpenFileName(self, "Open File", QDir.currentPath())
        self.content_image_source = fileName
        cont_img = cv2.imread(fileName)
        new_cont_img = self.adjust_image_size(cont_img)
        cv2.imwrite(CONTENT_IMAGE_NAME, new_cont_img)
        cont_seg = segment_this_img(CONTENT_IMAGE_NAME)
        cv2.imwrite(CONTENT_SEG_NAME,cont_seg)
        self.put_image(CONTENT_IMAGE_NAME, BASE_WIDTH, 0, 'Content image')
        self.put_image(BASE_BLACK_IMAGE_NAME, 2 * BASE_WIDTH, 0, '', False)

    def open_style(self):
        fileName, _ = QFileDialog.getOpenFileName(self, "Open File", QDir.currentPath())
        style_img = cv2.imread(fileName)
        new_style_img = self.adjust_image_size(style_img)
        cv2.imwrite(STYLE_IMAGE_NAME, new_style_img)
        style_seg = segment_this_img(STYLE_IMAGE_NAME)
        cv2.imwrite(STYLE_SEG_NAME,style_seg)
        self.put_image(STYLE_IMAGE_NAME, 0, 0, 'Style image')
        self.put_image(BASE_BLACK_IMAGE_NAME, 2*BASE_WIDTH, 0, '', False)

    def reset(self):
        self.buffer_image = np.zeros((IMAGE_HEIGHT,IMAGE_WIDTH), dtype=np.uint8)
        cv2.imwrite(BUFFER_IMAGE_NAME, self.buffer_image)
        image = QImage(BUFFER_IMAGE_NAME)
        self.imageLabel.setPixmap(QPixmap.fromImage(image))
        self.scaleFactor = 1.0
        self.fitToWindowAct.setEnabled(True)
        if not self.fitToWindowAct.isChecked():
            self.imageLabel.adjustSize()

    def transfer(self):
        process_stylization_ade20k.stylization(
            stylization_module=p_wct,
            smoothing_module=p_pro,
            content_image_path=CONTENT_IMAGE_NAME,
            style_image_path=STYLE_IMAGE_NAME,
            content_seg_path=CONTENT_SEG_NAME,
            style_seg_path=STYLE_SEG_NAME,
            output_image_path=SEG_STYLIZATION_OUTPUT_NAME,
            cuda=True,
            save_intermediate=args.save_intermediate,
            no_post=args.no_post,
            label_remapping=segReMapping
        )
        self.put_image(SEG_STYLIZATION_OUTPUT_NAME, 2*BASE_WIDTH, 0, 'Stylization Result')

    def put_image(self, fileName, xoff, yoff, image_name, boarder=True):
        if fileName:
            self.content_image_name = fileName
            self.content_image = cv2.imread(self.content_image_name)
            if self.content_image.shape[1] < self.content_image.shape[0]:
                R_HEIGHT = SMALL_BASE_WIDTH
                R_WIDTH = int( R_HEIGHT * self.content_image.shape[1]*1.0/self.content_image.shape[0] )
            else:
                R_WIDTH = SMALL_BASE_WIDTH
                R_HEIGHT = int( R_WIDTH * self.content_image.shape[0]*1.0/self.content_image.shape[1] )
            gap_x = int((BASE_WIDTH - R_WIDTH) / 2)
            gap_y = int((BASE_WIDTH - R_HEIGHT) / 2)
            display_img = cv2.resize(self.content_image, dsize=(R_WIDTH, R_HEIGHT))
            if boarder:
                display_img2 = cv2.copyMakeBorder(display_img,BORDER,BORDER,BORDER,BORDER,cv2.BORDER_CONSTANT,value=[255,255,255])
            else:
                display_img2 = cv2.resize(display_img, dsize=(display_img.shape[1]+2*BORDER, display_img.shape[0]+2*BORDER))
            self.buffer_image = cv2.imread(BUFFER_IMAGE_NAME)
            self.buffer_image[yoff:(yoff+BASE_WIDTH),xoff:(xoff+BASE_WIDTH),:] = 0
            nyoff = yoff-BORDER
            nxoff = xoff-BORDER
            self.buffer_image[(nyoff+gap_y):(nyoff+gap_y+R_HEIGHT+2*BORDER),
                              (nxoff+gap_x):(nxoff+gap_x+R_WIDTH+2*BORDER), :] = display_img2
            cv2.putText(self.buffer_image, image_name, (xoff+40,yoff+60), cv2.FONT_HERSHEY_SIMPLEX, 1.5, [0,255,0],2)
            cv2.imwrite(BUFFER_IMAGE_NAME, self.buffer_image)
            image = QImage(BUFFER_IMAGE_NAME)
            self.imageLabel.setPixmap(QPixmap.fromImage(image))
            self.fitToWindowAct.setEnabled(True)
            if not self.fitToWindowAct.isChecked():
                self.imageLabel.adjustSize()

    def normalSize(self):
        self.imageLabel.adjustSize()

    def fitToWindowAct(self):
        fitToWindowAct = self.fitToWindowAct.isChecked()
        self.scrollArea.setWidgetResizable(fitToWindowAct)
        if not fitToWindowAct:
            self.normalSize()

    def createActions(self):
        self.contentOpenAct = QAction("&Open", self, triggered=self.open_content)
        self.styleOpenAct = QAction("&Open", self, triggered=self.open_style)
        self.transferAct = QAction("&Transfer", self, triggered=self.transfer)
        self.resetAct = QAction("&Reset", self, triggered=self.reset)
        self.fitToWindowAct = QAction("&Fit to Window", self, enabled=False, checkable=True, shortcut="Ctrl+F", triggered=self.fitToWindowAct)


    def createMenus(self):
        self.contentMenu = QMenu("&Content", self)
        self.contentMenu.addAction(self.contentOpenAct)
        self.styleMenu = QMenu("&Style", self)
        self.styleMenu.addAction(self.styleOpenAct)
        self.transferMenu = QMenu("&Transfer", self)
        self.transferMenu.addAction(self.transferAct)
        self.resetMenu = QMenu("&Reset", self)
        self.resetMenu.addAction(self.resetAct)
        self.menuBar().addMenu(self.styleMenu)
        self.menuBar().addMenu(self.contentMenu)
        self.menuBar().addMenu(self.transferMenu)
        self.menuBar().addMenu(self.resetMenu)


    def adjustScrollBar(self, scrollBar, factor):
        scrollBar.setValue(int(factor * scrollBar.value() + ((factor - 1) * scrollBar.pageStep()/2)))



app = QApplication(sys.argv)
imageViewer = ImageViewer()
imageViewer.show()
sys.exit(app.exec_())