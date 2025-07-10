#!/usr/bin/python

'''
Function to determine foot's sidedeness in the image - for the foot-tracking app
by Rafal May 2025
''' 


# Set compute environment
import cv2
import csv
import math
import numpy as np
from datetime import datetime
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import get_bounding_box 


# Declarre reachable globally vars:
cnt1 = cnt2 = cx1 = cy1 = cx2 = cy2 = None


# The function
def tell_sidedeness(image, img_no, csv_file):
    # Tell left from right                
    side_R = False
    side_L = False

    # Sort out sidedeness for irregular predictions
    if  get_bounding_box.count < 3 or 2 < cv2.contourArea(get_bounding_box.cnt_1)/cv2.contourArea(get_bounding_box.cnt_2) or cv2.contourArea(get_bounding_box.cnt_1)/cv2.contourArea(get_bounding_box.cnt_2) < 0.5:    
        if get_bounding_box.get_bounding_box.count < 3:
            rect1 = cv2.minAreaRect(get_bounding_box.cnt_1)
        else: 
            rect1 = cv2.minAreaRect(get_bounding_box.cnt_1)
            get_bounding_box.cnt_1 = get_bounding_box.cnt_1 if cv2.contourArea(get_bounding_box.cnt_1) > cv2.contourArea(get_bounding_box.cnt_2) else get_bounding_box.cnt_2
            get_bounding_box.cx_1 = get_bounding_box.cx_1 if cv2.contourArea(get_bounding_box.cnt_1) != cv2.contourArea(get_bounding_box.cnt_2)  else get_bounding_box.cx_2
            get_bounding_box.cy_1 = get_bounding_box.cy_1 if cv2.contourArea(get_bounding_box.cnt_1) != cv2.contourArea(get_bounding_box.cnt_2)  else get_bounding_box.cy_2
        

        # Compute minimum area bounding box
        rect1 = cv2.minAreaRect(get_bounding_box.cnt_1)
        box1 = cv2.boxPoints(rect1)
        box1 = np.intp(box1)


        cx, cy = rect1[0]  # Center of the rectangle
        width, height = rect1[1]  # Box dimensions
        angle = rect1[2]  # Rotation angle


        # Determine longer axis direction
        longer_side_angle = angle if width > height else angle + 90
        longer_side_angle_rad = np.deg2rad(longer_side_angle)


        # Find the uppermost point
        uppermost_point = min(get_bounding_box.cnt_1, key=lambda point: (point[0][1], point[0][0]))
        # Unpack coordinates
        px, py = uppermost_point[0]  


        # Define a 20% segment along the longer axis
        segment_length = 0.2 * max(width, height)
        y_shift = segment_length * np.sin(longer_side_angle_rad)


        # Extract points in the 20% segment
        selected_points = [p[0] for p in get_bounding_box.cnt_1 if (py <= p[0][1] <= py + y_shift)]


        # Divide into left and right based on symmetry axis
        left_side = [p for p in selected_points if p[0] > cx]
        right_side = [p for p in selected_points if p[0] <= cx]


        # Compute center of mass for left and right
        if left_side:
            cx_left = sum(p[0] for p in left_side) / len(left_side)

        if right_side:
            cx_right = sum(p[0] for p in right_side) / len(right_side)


        # Determine sidedness based on mask location (lower x = right foot)
        if right_side and left_side:
            if cx_right < cx_left:
                # Right foot appears on the left side of the image
                side_L = True
                cnt2 = get_bounding_box.cnt_1
                cx2 = get_bounding_box.cx_1 
                cy2 = get_bounding_box.cy_1

            else:
                # Left foot appears on the right side of the image
                side_R = True
                cnt1 = get_bounding_box.cnt_1
                cx1 = get_bounding_box.cx_1 
                cy1 = get_bounding_box.cy_1


    # Sort out sidedeness for normally predicted two feet
    else:
        side_R = True
        side_L = True
        if get_bounding_box.cx_2 > get_bounding_box.cx_1 and get_bounding_box.cx_1 < 160:   
            cnt1 = get_bounding_box.cnt_2
            cnt2 = get_bounding_box.cnt_1
            cx1 = get_bounding_box.cx_2
            cy1 = get_bounding_box.cy_2
            cx2 = get_bounding_box.cx_1
            cy2 = get_bounding_box.cy_1

        else:
            cnt1 = get_bounding_box.cnt_1
            cnt2 = get_bounding_box.cnt_2
            cx1 = get_bounding_box.cx_1
            cy1 = get_bounding_box.cy_1
            cx2 = get_bounding_box.cx_2
            cy2 = get_bounding_box.cy_2        


    if side_R == True:
        if cv2.contourArea(cnt1) < 0.5 * old_area_cnt1 and get_bounding_box.old_cy_1 > 0: 
            cnt_1_area = 1
        elif cv2.contourArea(cnt1) < 2500 or cv2.contourArea(cnt1) < 0.5 * old_area_cnt2:         
            cnt_1_area = 1
        else:  
            old_area_cnt1 = cnt_1_area = cv2.contourArea(cnt1)

    if side_L == True:
        if (cv2.contourArea(cnt2) < 0.5 * old_area_cnt2 or cv2.contourArea(cnt2) > 2 * old_area_cnt2) and get_bounding_box.old_cy_1 > 0: 
            cnt_2_area = 1
        elif cv2.contourArea(cnt2) < 2500 or cv2.contourArea(cnt2) < 0.5 * old_area_cnt1:
            cnt_2_area = 1
        else:  
            old_area_cnt2 = cnt_2_area = cv2.contourArea(cnt2)
       

    pass1 = False
    pass2 = False


    if  get_bounding_box.count > 2 and cnt_1_area > 1 and cnt_2_area > 1:
        pass1 = pass2 = True
        print('all??')
        # Set left and right
    
    elif side_L == False or (cnt_1_area > 1 and cnt_2_area == 1): # or get_bounding_box.count < 3:
        pass1 = True
        print('Pass1??')
    
    elif get_bounding_box.count >= 3 and cnt_1_area == 1 and cnt_2_area == 1:
        temp_evolution =  np.zeros_like(cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)) 
        get_bounding_box.process_frame = False
        with open(csv_file, mode='a', newline='') as file2:
            writer = csv.writer(file2)
            # Append selected frame number to CSV file
            writer.writerow([img_no])
        return np.zeros_like(image), None, None
    
    elif side_L == False and cnt_1_area == 1:
        temp_evolution =  np.zeros_like(cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)) 
        get_bounding_box.process_frame = False
        with open(csv_file, mode='a', newline='') as file2:
            writer = csv.writer(file2)
            # Append selected frame number to CSV file
            writer.writerow([img_no])
        return np.zeros_like(image), None, None
    
    elif side_R == False or (cnt_2_area > 1 and cnt_1_area == 1):
        pass2 = True
        print('Pass2??')
    
    return image, pass1, pass2