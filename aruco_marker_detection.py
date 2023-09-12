"""
@author: valentin morel

Detection of the Aruco marker with the world camera
in order to gather data to train the SVR regression.
"""

from typing import Tuple

from WorldCameraOpening import WorldCameraFrame
import cv2
import cv2.aruco as aruco
import numpy as np

# Constant parameters used in Aruco methods
ARUCO_PARAMETERS = aruco.DetectorParameters_create()
ARUCO_DICT = aruco.Dictionary_get(aruco.DICT_6X6_1000)


def world_coord() -> Tuple[float, float, np.ndarray, bool]:
    """
    Detect Aruco marker and return its world coordinates along with the modified image and a flag indicating 
    whether the marker was successfully detected.

    Returns:
        Tuple containing the x and y world coordinates of the marker, the modified image with detected markers 
        outlined, and a flag indicating whether the marker was successfully detected.
    """
    
    # Capturing each frame of our video stream
    rgb_image = WorldCameraFrame()
    
    # Convert image to grayscale
    gray_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2GRAY)
    
    # Detect Aruco markers
    corners, ids, _ = aruco.detectMarkers(gray_image, ARUCO_DICT, parameters=ARUCO_PARAMETERS)
    
    flag_data = False
    cx_world, cy_world = 0.0, 0.0

    if isinstance(ids, np.ndarray):
        if ids.size == 1 and ids[0] == 0:
            cx_world = corners[0][0][0][0]
            cy_world = corners[0][0][0][1]
            #print('cx_world: ', cx_world)
            #print('cy_world: ', cy_world)
            flag_data = True

        # Outline all of the markers detected in our image
        rgb_image = aruco.drawDetectedMarkers(rgb_image, corners, ids, borderColor=(0, 0, 255))

    else:
        # Outline all of the markers detected in our image even if no valid marker was detected
        rgb_image = aruco.drawDetectedMarkers(rgb_image, corners, ids, borderColor=(0, 0, 255))
        flag_data = False

    return cx_world, cy_world, rgb_image, flag_data


if __name__ == '__main__':
    world_coord()
