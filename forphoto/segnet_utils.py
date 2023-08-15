#
# Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.
#
from jetson_utils import cudaAllocMapped, cudaToNumpy, cudaDeviceSynchronize
import numpy as np

class segmentationBuffers:
    def __init__(self, net, args):
        self.net = net
        self.overlay = None
        
        self.use_overlay = "overlay" in args.visualize
        
        if not self.use_overlay:
            raise Exception("Geçerli görselleştirme bayrakları arasında 'overlay' bulunmalıdır.")
        
        self.grid_width, self.grid_height = net.GetGridSize()
        self.num_classes = net.GetNumClasses()

    def createOverlay(self, img_input):
        if not self.overlay:
            shape = img_input.shape
            format = "rgb8"
            self.overlay = cudaAllocMapped(width=shape[1], height=shape[0], format=format)

        self.net.Process(img_input)
        self.net.Overlay(self.overlay)
        cudaDeviceSynchronize()

        return cudaToNumpy(self.overlay)




