import rospy
from sensor_msgs.msg import Image
import cv2
from cv_bridge import CvBridge

import jetson.inference
import jetson.utils

import numpy as np

class DepthProcessor:

    def __init__(self):
        # ROS Initialization
        rospy.init_node('depth_processing_node', anonymous=True)
        self.bridge = CvBridge()

        # Logging for debugging purposes
        rospy.loginfo("Starting DepthProcessor initialization...")

        # Jetson Inference Initialization
        self.net = jetson.inference.depthNet()
        rospy.loginfo("DepthNet initialized.")

        self.depth_field = self.net.GetDepthField()
        rospy.loginfo("DepthField retrieved.")

        rospy.sleep(1)  # Sleep for a short duration before subscribing to avoid potential race conditions
        self.image_subscriber = rospy.Subscriber('/webcam/image_raw', Image, self.image_callback)
        rospy.loginfo("Subscribed to /webcam/image_raw.")

        rospy.loginfo("DepthProcessor initialized successfully!")

    def image_callback(self, img_msg):
        if not hasattr(self, 'net'):
            rospy.logerr("Net attribute not initialized!")
            return

        # Convert ROS Image message to CV2 Image
        cv_image = self.bridge.imgmsg_to_cv2(img_msg, "bgr8")

        # Convert CV2 Image to CUDA Image
        cuda_image = jetson.utils.cudaFromNumpy(cv_image)

        # Process Image
        self.net.Process(cuda_image)
        jetson.utils.cudaDeviceSynchronize()  # Wait for GPU to finish processing

        # Update the depth_numpy after processing
        self.depth_numpy = jetson.utils.cudaToNumpy(self.depth_field)

        # Extract Depth Information
        min_depth = np.amin(self.depth_numpy)
        max_depth = np.amax(self.depth_numpy)

        rospy.loginfo(f"Min Depth: {min_depth}, Max Depth: {max_depth}")

        # Normalize depth for visualization
        normalized_depth = cv2.normalize(self.depth_numpy, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

        # Apply bilateral filter for noise reduction
        filtered_depth = cv2.bilateralFilter(normalized_depth, 5, 50, 50)

        # Apply colormap for better visualization
        colored_depth = cv2.applyColorMap(filtered_depth, cv2.COLORMAP_JET)

        # Resize the images for better visualization
        display_size = (800, 600)  # For example, resize to 800x600 pixels. You can adjust this value.
        resized_cv_image = cv2.resize(cv_image, display_size)
        resized_depth_image = cv2.resize(colored_depth, display_size)

        # Display the resized original image and the depth image
        cv2.imshow('Webcam Image', resized_cv_image)
        cv2.imshow('Depth Image', resized_depth_image)
        cv2.waitKey(1)

    def run(self):
        rospy.spin()

if __name__ == '__main__':
    depth_processor = DepthProcessor()
    depth_processor.run()

