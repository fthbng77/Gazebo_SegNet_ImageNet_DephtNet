# coding=utf-8
import cv2
import os
from jetson_inference import segNet
from jetson_utils import cudaFromNumpy, cudaDeviceSynchronize
from segnet_utils import segmentationBuffers  # <-- Yardımcı dosyayı burada ekliyoruz

class Args:
    def __init__(self):
        self.visualize = "overlay"

args = Args()
network = "fcn-resnet18-deepscene-576x320"

def process_image(img_input, buffers):
    img_input_cuda = cudaFromNumpy(img_input)
    return buffers.createOverlay(img_input_cuda)

def main():
    global net, buffers

    current_folder = os.getcwd()
    output_folder = "maskele"
    
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    net = segNet(network)
    buffers = segmentationBuffers(net, args)

    for image_name in os.listdir(current_folder):
        if image_name.endswith(".jpg"):
            image_path = os.path.join(current_folder, image_name)
            output_path = os.path.join(output_folder, image_name)

            img_input = cv2.imread(image_path)
            processed_image = process_image(img_input, buffers)
            processed_image = cv2.cvtColor(processed_image, cv2.COLOR_RGBA2BGR)
            cv2.imwrite(output_path, processed_image)

if __name__ == '__main__':
    main()

