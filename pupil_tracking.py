"""
@author: valentin morel

Track the pupil and the marker (ArucoMarkerDetection.py) to record data
in order to train a SVR (mySVR.py)
"""

import logging
import csv
import time

import cv2
import numpy as np

from eye_camera_opening import eye_camera_frame
from aruco_marker_detection import world_coord
from kernel_convolution import convolution

logging.basicConfig(level=logging.INFO)


def image_processing_eye(my_cap: np.ndarray, threshold: int) -> tuple:
    """
    Process the eye image to identify the pupil and its coordinates.

    Parameters:
    my_cap (np.ndarray): The captured image frame to be processed.
    threshold (int): Threshold value for image processing.

    Returns:
    tuple: Returns a tuple containing the coordinates of the pupil center (cx, cy) and processed images at different stages.
    """

    rgb_image = my_cap

    # Define the pupil Region Of Interest
    roi_x_min, roi_x_max, roi_y_min, roi_y_max = convolution(rgb_image)
    rgb_image_roi = rgb_image[roi_y_min:roi_y_max, roi_x_min:roi_x_max]
    
    # Image porcessing operations on the frame
    gray_image = cv2.cvtColor(rgb_image_roi, cv2.COLOR_BGR2GRAY)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (35, 35))
    open_image = cv2.morphologyEx(gray_image, cv2.MORPH_OPEN, kernel)
    
    ret, threshold_gray_image = cv2.threshold(open_image, threshold, 255, cv2.THRESH_BINARY_INV)
    median_blurred_image = cv2.medianBlur(threshold_gray_image, 7)
    blurred_gauss_image = cv2.GaussianBlur(threshold_gray_image, (5, 5), 0)

    my_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    opening_image = cv2.morphologyEx(blurred_gauss_image, cv2.MORPH_OPEN, my_kernel)

    
    nbr_ellipse = 0
    pi_4 = np.pi * 4
    old_circularity = -50
    old_area = -50

    # Find contours in the binary image after blurring and opening
    im2, contours, hierarchy = cv2.findContours(opening_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    pupil_contour = []

    for c in contours:
        area = cv2.contourArea(c)
        arc_len = cv2.arcLength(c, True)

        new_circularity = (pi_4 * area) / (arc_len * arc_len) if arc_len > 0 else -100

        if new_circularity > old_circularity:
            pupil_contour = c
            old_circularity = new_circularity
            old_area = area

    # Check if the selected contour can be the pupil. Multiple contours could be detected in the frame.
    if len(pupil_contour) > 4 and old_circularity > 0.55 and 6000 > old_area > 200:
        my_ellipse = cv2.fitEllipse(pupil_contour)
        nbr_ellipse += 1

    if nbr_ellipse == 1:
        coord_ellipse = my_ellipse[0]
        x, y = coord_ellipse
        cx = int(x) + roi_x_min
        cy = int(y) + roi_y_min

        # Fit ellipse on RGB Image                    
        cv2.ellipse(rgb_image_roi, my_ellipse, (0, 255, 255), 2, cv2.LINE_AA)
        cv2.circle(rgb_image, (cx, cy), 5, (0, 255, 255), -5)
    else:
        cx = 0
        cy = 0      

    return (cx, cy, nbr_ellipse, rgb_image, gray_image, opening_image, threshold_gray_image, median_blurred_image, blurred_gauss_image)


def detect_pupils_and_markers() -> None:
    """
    Continuously captures frames and processes them to identify and record pupil and marker coordinates.
    
    This function captures frames from different camera perspectives (left, right, world), processes them to identify 
    the pupil and marker coordinates, and saves this data to a CSV file. It also saves the processed video streams 
    to output video files.

    Returns:
    None: This function does not return anything.
    """
    
    # ID to determine if a lot of frames are skipped during the tracking of the pupil
    frame_id = 0  

    with open('pupil_coord.csv', 'w', newline='') as plc:
        my_writer = csv.writer(plc, delimiter=',')
        pupil_coord = [['ID', 'cx_left', 'cy_left', 'cx_right', 'cy_right', 'cx_world', 'cy_world']]
        my_writer.writerows(pupil_coord)

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out_left = cv2.VideoWriter('eye_left.avi', fourcc, 60, (400, 400))    
    out_right = cv2.VideoWriter('eye_right.avi', fourcc, 60, (400, 400))
    out_world = cv2.VideoWriter('world.avi', fourcc, 60, (1280, 720))
    
    while True:
        cap_left, cap_right = eye_camera_frame()
        
        # Note: Replace `threshold_value` with the appropriate value for thresholding
        threshold_value = 40

        # Processing the left eye image
        cx_left, cy_left, nbr_ellipse_left, rgb_image_left, _, _, _, _, _ = image_processing_eye(
            cap_left, threshold_value)

        # Processing the right eye image
        cx_right, cy_right, nbr_ellipse_right, rgb_image_right, _, _, _, _, _ = image_processing_eye(
            cap_right, threshold_value)
        
        # Getting World data from the ArucoMarkerDetection script to detect the ArucoMarker
        cx_world, cy_world, rgb_image_world, flag_data = world_coord()

        # Check if both pupils and the marker are detected otherwise skip the frame
        print(nbr_ellipse_left & nbr_ellipse_right & flag_data)
        if (nbr_ellipse_left & nbr_ellipse_right& flag_data) == 1:
            with open('pupil_coord.csv', 'a', newline='') as plc:
                my_writer = csv.writer(plc, delimiter=',')
                
                # Write pupil & world coordinates
                my_writer.writerow([frame_id, cx_left, cy_left, cx_right, cy_right, cx_world, cy_world])

        # Display the images
        cv2.imshow("Image Left", rgb_image_left)
        cv2.imshow("Image Right", rgb_image_right)
        cv2.imshow('Image World', rgb_image_world)

        # Write the output videos
        out_left.write(rgb_image_left)
        out_right.write(rgb_image_right)
        out_world.write(rgb_image_world)

        # Increment the ID for the next frame
        frame_id += 1

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the VideoWriter objects and destroy OpenCV windows
    out_left.release()
    out_right.release()
    out_world.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    detect_pupils_and_markers()
