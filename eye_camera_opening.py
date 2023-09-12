"""
@author: valentin morel

Open the two eye cameras with the UVC library
"""

import logging
from typing import Tuple

import cv2
import uvc

logging.basicConfig(level=logging.INFO)

# Initialize camera captures
dev_list = uvc.device_list()
cap_left = uvc.Capture(dev_list[0]['uid'])
cap_right = uvc.Capture(dev_list[1]['uid'])

# Configure the cameras
controls_dict_left = {c.display_name: c for c in cap_left.controls}
controls_dict_right = {c.display_name: c for c in cap_right.controls}

controls_dict_left['Auto Exposure Mode'].value = 1
controls_dict_right['Auto Exposure Mode'].value = 1

controls_dict_left['Gamma'].value = 200
controls_dict_right['Gamma'].value = 200

# Set resolution and FPS
cap_left.frame_mode = (400, 400, 60)
cap_right.frame_mode = (400, 400, 60)


def eye_camera_frame() -> Tuple[np.ndarray, np.ndarray]:
    """
    Capture frames from both the left and right eye cameras.

    Returns:
        Tuple of numpy arrays containing the captured frames for the left and right cameras.
    """
    frame_left = cap_left.get_frame_robust()
    rgb_image_left = frame_left.bgr

    frame_right = cap_right.get_frame_robust()
    rgb_image_right = frame_right.bgr

    return rgb_image_left, rgb_image_right


def display_camera_feed():
    """
    Test function to display the feed from both cameras.
    Press 'q' to exit the loop and end the feed.
    """
    # Uncomment the following line to display available modes for the right camera
    # print(cap_right.available_modes)
    
    while True:
        rgb_image_left, rgb_image_right = eye_camera_frame()
        
        cv2.imshow("Left Eye Camera", rgb_image_left)
        cv2.imshow("Right Eye Camera", rgb_image_right)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


if __name__ == '__main__':
    display_camera_feed()
