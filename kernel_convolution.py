"""
@author: valentin morel

Convolution with the eye frames in order to determine the ROI of the pupil
"""

from typing import Tuple

import cv2
import numpy as np
import matplotlib.pyplot as plt


def check_coord(coord: int) -> int:
    """Ensure the coordinate is within the valid range (0-399).

    Args:
        coord (int): The coordinate to check.

    Returns:
        int: The adjusted coordinate.
    """
    if coord < 0:
        return 0
    elif coord > 399:
        return 400
    else:
        return coord


def convolution(mycap: np.ndarray) -> Tuple[int, int, int, int]:
    """Determine the Region of Interest (ROI) of the pupil through convolution.

    Args:
        mycap (np.ndarray): The input image frame.

    Returns:
        tuple: The coordinates defining the ROI (roi_x_min, roi_x_max, roi_y_min, roi_y_max).
    """
    rgb_image = mycap
    gray_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2GRAY)

    a = 1.1
    kernel = np.array([
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0, a, a, 0, 0],
        [0, 0, a, a, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0]
    ], np.float32)

    # Convolution of the input gray image with the kernel to detect pupil ROI    
    output = cv2.filter2D(gray_image, -1, kernel)

    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(output)

    x_min = min_loc[0]
    y_min = min_loc[1]
    my_range = 100

    # Define the pupil ROI
    roi_x_min = x_min - my_range
    roi_x_max = x_min + my_range
    roi_y_min = y_min - my_range
    roi_y_max = y_min + my_range

    # Check if the coordinates are not outside the size of the image (400x400)
    roi_x_min = check_coord(roi_x_min)
    roi_x_max = check_coord(roi_x_max)
    roi_y_min = check_coord(roi_y_min)
    roi_y_max = check_coord(roi_y_max)

    return roi_x_min, roi_x_max, roi_y_min, roi_y_max


