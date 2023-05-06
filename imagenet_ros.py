#!/usr/bin/env python3
from jetson_inference import imageNet
import rospy
from sensor_msgs.msg import Image
from std_msgs.msg import String
from jetson_utils import cudaFromNumpy
from cv_bridge import CvBridge
import cv2

def image_callback(msg):
    global cv_img
    cv_img = CvBridge().imgmsg_to_cv2(msg, desired_encoding="passthrough")
    cuda_img = cudaFromNumpy(cv_img)

    # classify the image
    class_idx, confidence = net.Classify(cuda_img)

    # find the object description
    class_desc = net.GetClassDesc(class_idx)

    # publish the result
    pub.publish(class_desc)

    # display the result on the image window
    font = cv2.FONT_HERSHEY_SIMPLEX
    text = "{}: {:.2f}%".format(class_desc, confidence * 100)
    cv2.putText(cv_img, text, (10, 30), font, 1, (255, 255, 255), 2)

    rospy.loginfo("image is recognized as '{:s}' (class #{:d}) with {:f}% confidence".format(class_desc, class_idx, confidence * 100))

def main():
    global net, pub, cv_img

    net = imageNet("googlenet")

    rospy.init_node('imagenet_node', anonymous=True)

    sub = rospy.Subscriber('/webcam/image_raw', Image, image_callback)

    pub = rospy.Publisher('/imagenet_result', String, queue_size=10)

    # create a window to display the image
    cv2.namedWindow("Image", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Image", 640, 480)
    
    while not rospy.is_shutdown():
        if cv_img is not None:
            # display the image
            cv2.imshow("Image", cv_img)
        cv2.waitKey(1)

    cv2.destroyAllWindows()

if __name__ == '__main__':
    cv_img = None
    main()
