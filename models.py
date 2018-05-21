"""
Copyright (C) 2018 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
import torch.nn as nn


class VGGEncoder1(nn.Module):
    def __init__(self):
        super(VGGEncoder1, self).__init__()
        self.level = 1       
        # 224 x 224
        self.conv0 = nn.Conv2d(3, 3, 1, 1, 0)      
        self.pad1_1 = nn.ReflectionPad2d((1, 1, 1, 1))
        # 226 x 226
        self.conv1_1 = nn.Conv2d(3, 64, 3, 1, 0)
        self.relu1_1 = nn.ReLU(inplace=True)
        
    def forward1(self, x):
        return self.relu1_1(self.conv1_1(self.pad1_1(self.conv0(x))))
        
    def forward(self, x):
        return self.forward1(x)
        
class VGGEncoder2(VGGEncoder1):
    def __init__(self):
        super(VGGEncoder2, self).__init__()
        self.level = 2

        # 224 x 224        
        self.pad1_2 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv1_2 = nn.Conv2d(64, 64, 3, 1, 0)
        self.relu1_2 = nn.ReLU(inplace=True)
        # 224 x 224
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)
        # 112 x 112

        self.pad2_1 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv2_1 = nn.Conv2d(64, 128, 3, 1, 0)
        self.relu2_1 = nn.ReLU(inplace=True)
        # 112 x 112

    def forward2(self, x):
        out = self.pad1_2(x)
        pool1 = self.relu1_2(self.conv1_2(out))
        out, pool1_idx = self.maxpool1(pool1)
        out = self.pad2_1(out)
        out = self.conv2_1(out)
        out = self.relu2_1(out)
        return out, pool1_idx, pool1.size()

    def forward(self, x):
        out1 = self.forward1(x)
        return self.forward2(out1)
        
class VGGEncoder3(VGGEncoder2):
    def __init__(self):
        super(VGGEncoder3, self).__init__()
        self.level = 3
        
        self.pad2_2 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv2_2 = nn.Conv2d(128, 128, 3, 1, 0)
        self.relu2_2 = nn.ReLU(inplace=True)
        # 112 x 112
        
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)
        # 56 x 56
        
        self.pad3_1 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv3_1 = nn.Conv2d(128, 256, 3, 1, 0)
        self.relu3_1 = nn.ReLU(inplace=True)
        # 56 x 56

    def forward3(self, x):
        out = self.pad2_2(x)
        out = self.conv2_2(out)
        pool2 = self.relu2_2(out)
        
        out, pool2_idx = self.maxpool2(pool2)
        
        out = self.pad3_1(out)
        out = self.conv3_1(out)
        out = self.relu3_1(out)
        return out, pool2_idx, pool2.size()

    def forward(self, x):
        out1 = self.forward1(x)
        out2, pool1_idx, pool1_size = self.forward2(out1)
        out3, pool2_idx, pool2_size = self.forward3(out2)
        return out3, pool1_idx, pool1_size, pool2_idx, pool2_size 
        
class VGGEncoder4(VGGEncoder3):
    def __init__(self):
        super(VGGEncoder4, self).__init__()
        self.pad3_2 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv3_2 = nn.Conv2d(256, 256, 3, 1, 0)
        self.relu3_2 = nn.ReLU(inplace=True)
        # 56 x 56
        
        self.pad3_3 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv3_3 = nn.Conv2d(256, 256, 3, 1, 0)
        self.relu3_3 = nn.ReLU(inplace=True)
        # 56 x 56
        
        self.pad3_4 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv3_4 = nn.Conv2d(256, 256, 3, 1, 0)
        self.relu3_4 = nn.ReLU(inplace=True)
        # 56 x 56
        
        self.maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)
        # 28 x 28
        
        self.pad4_1 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv4_1 = nn.Conv2d(256, 512, 3, 1, 0)
        self.relu4_1 = nn.ReLU(inplace=True)
        # 28 x 28
    
    def forward4(self, x):
        out = self.pad3_2(x)
        out = self.conv3_2(out)
        out = self.relu3_2(out)
        
        out = self.pad3_3(out)
        out = self.conv3_3(out)
        out = self.relu3_3(out)
        
        out = self.pad3_4(out)
        out = self.conv3_4(out)
        pool3 = self.relu3_4(out)
        out, pool3_idx = self.maxpool3(pool3)
        
        out = self.pad4_1(out)
        out = self.conv4_1(out)
        out = self.relu4_1(out)
        
        return out, pool3_idx, pool3.size()
    
    def forward(self, x):
        out = self.forward1(x)
        out, pool1_idx, pool1_size = self.forward2(out)
        out, pool2_idx, pool2_size = self.forward3(out)
        out, pool3_idx, pool3_size = self.forward4(out)
        return out, pool1_idx, pool1_size, pool2_idx, pool2_size, pool3_idx, pool3_size

    
class VGGEncoder4M(VGGEncoder4):
    def __init__(self):
        super(VGGEncoder4M, self).__init__()
    
    def forward(self, x):
        out1 = self.forward1(x)
        out2, pool1_idx, pool1_size  = self.forward2(out1)
        out3, pool2_idx, pool2_size  = self.forward3(out2)
        out4, pool3_idx, pool3_size  = self.forward4(out3)
        return out4, out3, out2, out1

class VGGDecoder1(nn.Module):
    def __init__(self):
        super(VGGDecoder1, self).__init__()
        self.level = 1
        self.pad1_1 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv1_1 = nn.Conv2d(64, 3, 3, 1, 0)
    
    def forward(self, x):
        return  self.conv1_1(self.pad1_1(x))

            
class VGGDecoder2(VGGDecoder1):
    def __init__(self):
        super(VGGDecoder2, self).__init__()
        self.level = 2            
        self.pad2_1 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv2_1 = nn.Conv2d(128, 64, 3, 1, 0)
        self.relu2_1 = nn.ReLU(inplace=True)
        # 112 x 112
        
        self.unpool1 = nn.MaxUnpool2d(kernel_size=2, stride=2)
        # 224 x 224
        
        self.pad1_2 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv1_2 = nn.Conv2d(64, 64, 3, 1, 0)
        self.relu1_2 = nn.ReLU(inplace=True)
        # 224 x 224

    def forward(self, x, pool1_idx, pool1_size):
        out = self.pad2_1(x)
        out = self.conv2_1(out)
        out = self.relu2_1(out)
        out = self.unpool1(out, pool1_idx, output_size=pool1_size)
        
        out = self.pad1_2(out)
        out = self.conv1_2(out)
        out = self.relu1_2(out)
        return super(VGGDecoder2, self).forward(out)

class VGGDecoder3(VGGDecoder2):
    def __init__(self):
        super(VGGDecoder3, self).__init__()        
        self.level = 3
        self.pad3_1 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv3_1 = nn.Conv2d(256, 128, 3, 1, 0)
        self.relu3_1 = nn.ReLU(inplace=True)
        # 56 x 56
        
        self.unpool2 = nn.MaxUnpool2d(kernel_size=2, stride=2)
        # 112 x 112
        
        self.pad2_2 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv2_2 = nn.Conv2d(128, 128, 3, 1, 0)
        self.relu2_2 = nn.ReLU(inplace=True)
        # 112 x 112
        
    def forward(self, x, pool1_idx, pool1_size, pool2_idx, pool2_size):
        out = self.pad3_1(x)
        out = self.conv3_1(out)
        out = self.relu3_1(out)
        out = self.unpool2(out, pool2_idx, output_size=pool2_size)
        
        out = self.pad2_2(out)
        out = self.conv2_2(out)
        out = self.relu2_2(out)
        return super(VGGDecoder3, self).forward(out, pool1_idx, pool1_size)
            
class VGGDecoder4(VGGDecoder3):
    def __init__(self):
        super(VGGDecoder4, self).__init__()        

        self.level = 4
        self.pad4_1 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv4_1 = nn.Conv2d(512, 256, 3, 1, 0)
        self.relu4_1 = nn.ReLU(inplace=True)
        # 28 x 28
    
        self.unpool3 = nn.MaxUnpool2d(kernel_size=2, stride=2)
        # 56 x 56
        
        self.pad3_4 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv3_4 = nn.Conv2d(256, 256, 3, 1, 0)
        self.relu3_4 = nn.ReLU(inplace=True)
        # 56 x 56
        
        self.pad3_3 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv3_3 = nn.Conv2d(256, 256, 3, 1, 0)
        self.relu3_3 = nn.ReLU(inplace=True)
        # 56 x 56
        
        self.pad3_2 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv3_2 = nn.Conv2d(256, 256, 3, 1, 0)
        self.relu3_2 = nn.ReLU(inplace=True)
        # 56 x 56
        
    def forward(self, x, pool1_idx, pool1_size, pool2_idx, pool2_size, pool3_idx,
                pool3_size):
        out = self.pad4_1(x)
        out = self.conv4_1(out)
        out = self.relu4_1(out)
        out = self.unpool3(out, pool3_idx, output_size=pool3_size)
        
        out = self.pad3_4(out)
        out = self.conv3_4(out)
        out = self.relu3_4(out)
        
        out = self.pad3_3(out)
        out = self.conv3_3(out)
        out = self.relu3_3(out)
        
        out = self.pad3_2(out)
        out = self.conv3_2(out)
        out = self.relu3_2(out)
        return super(VGGDecoder4, self).forward(out, pool1_idx, pool1_size, pool2_idx, pool2_size)
        
