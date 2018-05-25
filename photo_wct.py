"""
Copyright (C) 2018 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import models

class PhotoWCT(nn.Module):
    def __init__(self, opt):
        super(PhotoWCT, self).__init__()
        self.opt = opt
        self.e1 = models.VGGEncoder1()
        self.d1 = models.VGGDecoder1()
        self.e2 = models.VGGEncoder2()
        self.d2 = models.VGGDecoder2()
        self.e3 = models.VGGEncoder3()
        self.d3 = models.VGGDecoder3()
        self.e4 = models.VGGEncoder4()
        # boris: This node is needed to unwrap what used to be 2 calls for the same layer
        # (one forward() and one forward_multiple())
        self.e4M = models.VGGEncoder4M()
        self.d4 = models.VGGDecoder4()
        
    def load (self):
        try:
            pretrained_dict = torch.load(self.opt.model)
            model_dict = self.state_dict()

            # Modify to match our changes
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
            model_dict.update(pretrained_dict)
#            print pretrained_dict.keys()
#            print model_dict.keys()
            # need to modify saved dict due to new eM4 node I have added
            for name, param in pretrained_dict.items():             
#                print('Checking ', name)
                if name.startswith('e4'):
                    mname = name.replace('e4.', 'e4M.')
                    if mname in model_dict:
                        print('Duplicating', name, mname)
                        pretrained_dict[mname] = param

            print('Checking keys ... ')
            missing = set(model_dict.keys()) - set(pretrained_dict.keys())
            if len(missing) > 0:
                err = 'missing keys in state_dict: "{}"'.format(missing)
                print ('Error: ',  err)

            extra = set(pretrained_dict.keys()) - set(model_dict.keys())
            if len(extra) > 0:
                err = 'extra keys in state_dict: "{}"'.format(extra)
                print('Error: ', err)
                
            print('Loading state ')

            self.load_state_dict(pretrained_dict)
            print('Loaded state ')
            return True
        except :
            print("Fail to load PhotoWCT models. PhotoWCT submodule not updated?")
            return False

    def pre_processing(self, cont_img, styl_img, cont_seg, styl_seg):
        self.__compute_label_info(cont_seg, styl_seg)

        cont_w = cont_seg.shape[1]
        cont_h = cont_seg.shape[0]
        styl_w = styl_seg.shape[1]
        styl_h = styl_seg.shape[0]

        cont_w1 = cont_w - 1
        cont_h1 = cont_h - 1
        cont_w2 = int(np.ceil(cont_w/2)-1)
        cont_h2 = int(np.ceil(cont_h/2)-1)
        cont_w3 = int(np.ceil((cont_w2+1)/2)-1)
        cont_h3 = int(np.ceil((cont_h2+1)/2)-1)
        cont_w4 = int(np.ceil((cont_w3+1)/2)-1)
        cont_h4 = int(np.ceil((cont_h3+1)/2)-1)
        
        styl_w1 = styl_w - 1
        styl_h1 = styl_h - 1
        styl_w2 = int(np.ceil(styl_w/2)-1)
        styl_h2 = int(np.ceil(styl_h/2)-1)
        styl_w3 = int(np.ceil((styl_w2+1)/2)-1)
        styl_h3 = int(np.ceil((styl_h2+1)/2)-1)
        styl_w4 = int(np.ceil((styl_w3+1)/2)-1)
        styl_h4 = int(np.ceil((styl_h3+1)/2)-1)
        
        self.cont_indi4, self.styl_indi4 = self.__compute_mask(cont_seg, styl_seg, cont_w4, cont_h4, styl_w4, styl_h4)
        self.cont_indi3, self.styl_indi3 = self.__compute_mask(cont_seg, styl_seg, cont_w3, cont_h3, styl_w3, styl_h3)
        self.cont_indi2, self.styl_indi2 = self.__compute_mask(cont_seg, styl_seg, cont_w2, cont_h2, styl_w2, styl_h2)
        self.cont_indi1, self.styl_indi1 = self.__compute_mask(cont_seg, styl_seg, cont_w1, cont_h1, styl_w1, styl_h1)

    def do_processing(self, args):
        cont_img, styl_img, cont_seg, styl_seg = args
        self.pre_processing(cont_img, styl_img, cont_seg, styl_seg)
        return self.forward([cont_img, styl_img])
        
    def forward(self, args):
        cont_img, styl_img = args
        sF4, sF3, sF2, sF1 = self.e4M(styl_img)
        cF4, cpool_idx, cpool1, cpool_idx2, cpool2, cpool_idx3, cpool3 = self.e4(cont_img)
        sF4 = sF4.data.squeeze(0)
        cF4 = cF4.data.squeeze(0)

        csF4 = self.__feature_wct(cF4, sF4, self.cont_indi4, self.styl_indi4)
        Im4 = self.d4(csF4, cpool_idx, cpool1, cpool_idx2, cpool2, cpool_idx3, cpool3)

        # boris: if you uncomment next line, it would export correctly (at least the graph would look as expected)

        # boris: next line is enough to break ONNX export
            
        cF3, cpool_idx, cpool1, cpool_idx2, cpool2 = self.e3(Im4)
    
        sF3 = sF3.data.squeeze(0)
        cF3 = cF3.data.squeeze(0)
        csF3 = self.__feature_wct(cF3, sF3, self.cont_indi3, self.styl_indi3)
        Im3 = self.d3(csF3, cpool_idx, cpool1, cpool_idx2, cpool2)
        
        cF2, cpool_idx, cpool = self.e2(Im3)
        sF2 = sF2.data.squeeze(0)
        cF2 = cF2.data.squeeze(0)
        csF2 = self.__feature_wct(cF2, sF2, self.cont_indi2, self.styl_indi2)
        Im2 = self.d2(csF2, cpool_idx, cpool)
        
        cF1 = self.e1(Im2)
        sF1 = sF1.data.squeeze(0)
        cF1 = cF1.data.squeeze(0)
        csF1 = self.__feature_wct(cF1, sF1, self.cont_indi1, self.styl_indi1)
        Im1 = self.d1(csF1)
        return Im1
 
    def __compute_label_info(self, cont_seg, styl_seg):
        max_label = np.max(cont_seg) + 1
        self.label_set = np.unique(cont_seg)
        self.label_indicator = np.zeros(int(max_label))
        for l in self.label_set:
            # if l==0:
            #   continue
            is_valid = lambda a, b: a > 10 and b > 10 and a / b < 100 and b / a < 100
            o_cont_mask = np.where(cont_seg.reshape(cont_seg.shape[0] * cont_seg.shape[1]) == l)
            o_styl_mask = np.where(styl_seg.reshape(styl_seg.shape[0] * styl_seg.shape[1]) == l)
            self.label_indicator[int(l)] = is_valid(o_cont_mask[0].size, o_styl_mask[0].size)
    

    def __compute_mask(self, cont_seg, styl_seg, cont_w, cont_h, styl_w, styl_h):
        if True:
            if len(cont_seg.shape) == 2:
                t_cont_seg = np.asarray(Image.fromarray(cont_seg).resize((cont_w, cont_h), Image.NEAREST))
            else:
                t_cont_seg = np.asarray(Image.fromarray(cont_seg, mode='RGB').resize((cont_w, cont_h), Image.NEAREST))
            if len(styl_seg.shape) == 2:
                t_styl_seg = np.asarray(Image.fromarray(styl_seg).resize((styl_w, styl_h), Image.NEAREST))
            else:
                t_styl_seg = np.asarray(Image.fromarray(styl_seg, mode='RGB').resize((styl_w, styl_h), Image.NEAREST))
            
            for l in self.label_set:
                if self.label_indicator[l] == 0:
                    continue
                cont_mask = np.where(t_cont_seg.reshape(t_cont_seg.shape[0] * t_cont_seg.shape[1]) == l)
                styl_mask = np.where(t_styl_seg.reshape(t_styl_seg.shape[0] * t_styl_seg.shape[1]) == l)
                if cont_mask[0].size <= 0 or styl_mask[0].size <= 0:
                    continue                

        cont_indi = torch.LongTensor(cont_mask[0])
        styl_indi = torch.LongTensor(styl_mask[0])
        return cont_indi.cuda(), styl_indi.cuda()
                    
    def __feature_wct(self, cont_feat, styl_feat, cont_indi, styl_indi):
        cont_c, cont_h, cont_w = cont_feat.size(0), cont_feat.size(1), cont_feat.size(2)
        styl_c, styl_h, styl_w = styl_feat.size(0), styl_feat.size(1), styl_feat.size(2)
        
        cont_feat_view = cont_feat.view(cont_c, -1).clone()
        styl_feat_view = styl_feat.view(styl_c, -1).clone()
        target_feature = cont_feat.view(cont_c, -1).clone()
        
        cFFG = torch.index_select(cont_feat_view, 1, cont_indi)
        sFFG = torch.index_select(styl_feat_view, 1, styl_indi)
        # print(len(cont_indi))
        # print(len(styl_indi))
        tmp_target_feature = self.__wct_core(cFFG, sFFG)
        # print(tmp_target_feature.size())
        if True: # torch.__version__ >= "0.4.0":
            # This seems to be a bug in PyTorch 0.4.0 to me.
            new_target_feature = torch.transpose(target_feature, 1, 0)
            new_target_feature.index_copy_(0, cont_indi, \
                                           torch.transpose(tmp_target_feature,1,0))
            target_feature = torch.transpose(new_target_feature, 1, 0)
#        else:
#            target_feature.index_copy_(1, cont_indi, tmp_target_feature)
            
        target_feature = target_feature.view_as(cont_feat)
        ccsF = target_feature.float().unsqueeze(0)
        return ccsF
    
    def __wct_core(self, cont_feat, styl_feat):
        cFSize = cont_feat.size()
        c_mean = torch.mean(cont_feat, 1)  # c x (h x w)
        c_mean = c_mean.unsqueeze(1).expand_as(cont_feat)
        cont_feat = cont_feat - c_mean
        
        iden = torch.eye(cFSize[0])  # .double()
        if self.is_cuda:
            iden = iden.cuda()
        
        contentConv = torch.mm(cont_feat, cont_feat.t()).div(cFSize[1] - 1) + iden
        # del iden
        c_u, c_e, c_v = torch.svd(contentConv, some=False)
        # c_e2, c_v = torch.eig(contentConv, True)
        # c_e = c_e2[:,0]
        
        k_c = cFSize[0]
        for i in range(cFSize[0] - 1, -1, -1):
            if c_e[i] >= 0.00001:
                k_c = i + 1
                break
        
        sFSize = styl_feat.size()
        s_mean = torch.mean(styl_feat, 1)
        styl_feat = styl_feat - s_mean.unsqueeze(1).expand_as(styl_feat)
        styleConv = torch.mm(styl_feat, styl_feat.t()).div(sFSize[1] - 1)
        s_u, s_e, s_v = torch.svd(styleConv, some=False)
        
        k_s = sFSize[0]
        for i in range(sFSize[0] - 1, -1, -1):
            if s_e[i] >= 0.00001:
                k_s = i + 1
                break
        
        c_d = (c_e[0:k_c]).pow(-0.5)
        step1 = torch.mm(c_v[:, 0:k_c], torch.diag(c_d))
        step2 = torch.mm(step1, (c_v[:, 0:k_c].t()))
        whiten_cF = torch.mm(step2, cont_feat)
        
        s_d = (s_e[0:k_s]).pow(0.5)
        targetFeature = torch.mm(torch.mm(torch.mm(s_v[:, 0:k_s], torch.diag(s_d)), (s_v[:, 0:k_s].t())), whiten_cF)
        targetFeature = targetFeature + s_mean.unsqueeze(1).expand_as(targetFeature)
        return targetFeature
    
    @property
    def is_cuda(self):
        return next(self.parameters()).is_cuda
