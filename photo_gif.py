"""
Copyright (C) 2018 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
from __future__ import division
import torchvision.transforms as transforms
from PIL import Image
import torch
from torch import nn
from torch.autograd import Variable
import numpy as np
# import cv2
# from cv2.ximgproc import guidedFilter


class GIFSmoothing(nn.Module):
  def __init__(self, r, eps):
    super(GIFSmoothing, self).__init__()
    self.r = r
    self.eps = eps

  # def process(self, initImg, contentImg):
  #   '''
  #   :param initImg: intermediate output. Either image path or PIL Image
  #   :param contentImg: content image output. Either path or PIL Image
  #   :return: stylized output image. PIL Image
  #   '''
  #   if type(initImg) == str:
  #     init_img = cv2.imread(initImg)
  #     init_img = init_img[2:-2,2:-2,:]
  #   else:
  #     init_img = np.array(initImg)[:, :, ::-1].copy()
  #
  #   if type(contentImg) == str:
  #     cont_img = cv2.imread(contentImg)
  #   else:
  #     cont_img = np.array(contentImg)[:, :, ::-1].copy()
  #
  #   output_img = guidedFilter(guide=cont_img, src=init_img, radius=self.r, eps=self.eps)
  #   output_img = cv2.cvtColor(output_img, cv2.COLOR_BGR2RGB)
  #   output_img = Image.fromarray(output_img)
  #   return output_img

  def process(self, initImg, contentImg):
    '''
    PYTORCH GIF
    :param initImg: intermediate output. Either image path or PIL Image
    :param contentImg: content image output. Either path or PIL Image
    :return: stylized output image. PIL Image
    '''
    if type(initImg) == str:
      init_img = Image.open(initImg).convert('RGB')
      init_img = transforms.ToTensor()(init_img).unsqueeze(0)
    else:
      init_img = transforms.ToTensor()(initImg).unsqueeze(0)

    if type(contentImg) == str:
      cont_img = Image.open(contentImg).convert('RGB')
      cont_img = transforms.ToTensor()(cont_img).unsqueeze(0)
    else:
      cont_img = transforms.ToTensor()(contentImg).unsqueeze(0)

    cont_img = cont_img.cuda()
    init_img2 = init_img.cuda()
    cont_img = Variable(cont_img, volatile=True)
    init_img2 = Variable(init_img2, volatile=True)
    output_img = GuidedFilter(r=35,eps=0.001)(cont_img,init_img2)
    output_img = torch.clamp(output_img,0,1)
    output_img = transforms.ToPILImage()(output_img[0,:,:,:].data.cpu())
    return output_img


#
# Code below is duplicated from https://github.com/wuhuikai/DeepGuidedFilter
#

def diff_x(input, r):
  assert input.dim() == 4

  left = input[:, :, r:2 * r + 1]
  middle = input[:, :, 2 * r + 1:] - input[:, :, :-2 * r - 1]
  right = input[:, :, -1:] - input[:, :, -2 * r - 1:    -r - 1]

  output = torch.cat([left, middle, right], dim=2)

  return output


def diff_y(input, r):
  assert input.dim() == 4

  left = input[:, :, :, r:2 * r + 1]
  middle = input[:, :, :, 2 * r + 1:] - input[:, :, :, :-2 * r - 1]
  right = input[:, :, :, -1:] - input[:, :, :, -2 * r - 1:    -r - 1]

  output = torch.cat([left, middle, right], dim=3)

  return output


class BoxFilter(nn.Module):
  def __init__(self, r):
    super(BoxFilter, self).__init__()

    self.r = r

  def forward(self, x):
    assert x.dim() == 4

    return diff_y(diff_x(x.cumsum(dim=2), self.r).cumsum(dim=3), self.r)


class GuidedFilter(nn.Module):
    def __init__(self, r, eps=1e-8):
        super(GuidedFilter, self).__init__()

        self.r = r
        self.eps = eps
        self.boxfilter = BoxFilter(r)


    def forward(self, x, y):
        n_x, c_x, h_x, w_x = x.size()
        n_y, c_y, h_y, w_y = y.size()

        assert n_x == n_y
        assert c_x == 1 or c_x == c_y
        assert h_x == h_y and w_x == w_y
        assert h_x > 2 * self.r + 1 and w_x > 2 * self.r + 1

        # N
        N = self.boxfilter(Variable(x.data.new().resize_((1, 1, h_x, w_x)).fill_(1.0)))

        # mean_x
        mean_x = self.boxfilter(x) / N
        # mean_y
        mean_y = self.boxfilter(y) / N
        # cov_xy
        cov_xy = self.boxfilter(x * y) / N - mean_x * mean_y
        # var_x
        var_x = self.boxfilter(x * x) / N - mean_x * mean_x

        # A
        A = cov_xy / (var_x + self.eps)
        # b
        b = mean_y - A * mean_x

        # mean_A; mean_b
        mean_A = self.boxfilter(A) / N
        mean_b = self.boxfilter(b) / N

        return mean_A * x + mean_b



