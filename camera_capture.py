import pickle

import rospy
import numpy as np
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

bridge = CvBridge()
img_id = 0

def cam_listener(data):
    global img_id
    print(dir(data))
    print(data.height, data.width)
    img = bridge.imgmsg_to_cv2(data, 'bgr8').astype(np.ndarray)
    with open('img{}.txt'.format(img_id), 'wb') as f:
        pickle.dump(img, f)
    img_id += 1

if __name__ == '__main__':
    rospy.init_node('cam_listener')
    rospy.Subscriber('/camera/rgb/image_color', Image, cam_listener)
    rospy.spin()
