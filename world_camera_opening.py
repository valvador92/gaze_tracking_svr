"""
@author: valentin morel

Open the world camera with the uvc library
"""

from typing import Any
import cv2
import uvc
import logging
import numpy as np
import os

logging.basicConfig(level=logging.INFO)

dev_list = uvc.device_list()
cap_world = uvc.Capture(dev_list[2]['uid'])

current_path = os.path.dirname(os.path.abspath(__file__))

# Resolution, FPS
cap_world.frame_mode = (1280, 720, 60)

# Load the intrinsic matrix and distortion coefficients to undistort the image
camera_mtx = np.load(os.path.join(current_path, 'mtx.npy'))
dist_coefs = np.load(os.path.join(current_path, 'dist.npy'))


def world_camera_frame() -> Any:
    """
    Capture and undistort a frame from the world camera.

    Returns:
        The undistorted frame.
    """
    frame = cap_world.get_frame_robust()
    rgb_image = frame.bgr

    undst = cv2.undistort(rgb_image, camera_mtx, dist_coefs)

    return undst


def main_test() -> None:
    """
    Main test function to demonstrate capturing and displaying frames
    from the world camera in a loop until 'q' is pressed.
    """
    # Uncomment the following to know which mode one can chose for the camera
    # print(cap_world.available_modes)

    while True:
        undst = world_camera_frame()
        cv2.imshow("undst", undst)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


if __name__ == '__main__':
    main_test()
