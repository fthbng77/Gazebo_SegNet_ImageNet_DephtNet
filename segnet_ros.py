#!/usr/bin/env python3

import rospy
from sensor_msgs.msg import Image
from jetson_inference import segNet
from jetson_utils import videoOutput, cudaOverlay, cudaDeviceSynchronize, Log
from segnet_utils import *
import ros_numpy
from jetson_utils import cudaFromNumpy, cudaToNumpy
network = "fcn-resnet18-deepscene-576x320"
alpha = 150
output_path = "output.mp4"

rospy.init_node('segnet_node')
net = segNet(network)
net.SetOverlayAlpha(alpha)
output = videoOutput(output_path)

buffers = segmentationBuffers(net)
img_format = None

def image_callback(data):
    global img_format
    img_input = ros_numpy.numpify(data)
    if img_format is None:
        img_format = data.encoding
        buffers.Alloc(img_input.shape, img_format)
    else:
        img_input = cudaFromNumpy(img_input)
    net.Process(img_input)
    if buffers.overlay:
        net.Overlay(buffers.overlay)
    if buffers.mask:
        net.Mask(buffers.mask)
    if buffers.composite:
        cudaOverlay(buffers.overlay, buffers.composite, 0, 0)
        cudaOverlay(buffers.mask, buffers.composite, buffers.overlay.width, 0)
    output.Render(buffers.output)
    output.SetStatus("{:s} | Network {:.0f} FPS".format(network, net.GetNetworkFPS()))
    cudaDeviceSynchronize()
    net.PrintProfilerTimes()
    buffers.ComputeStats()
    pub.publish(ros_numpy.msgify(Image, cudaToNumpy(buffers.output), img_format))

pub = rospy.Publisher('/segnet_image', Image, queue_size=10)
rospy.Subscriber("/webcam/image_raw", Image, image_callback)
rospy.spin()
