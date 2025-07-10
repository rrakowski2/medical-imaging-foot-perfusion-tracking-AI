#!/usr/bin/python


'''
Script to get percentage-ratio isotherm area maps - for the foot-tracking app
by Rafal June 2025
''' 


# Prepare compute environment
import cv2, numpy as np, math


rounded_max_value = None
white_ratios = []
yellow_ratios = []
orange_ratios = []
red_ratios = []
blue_ratios = []


def get_percentage_maps(temp_image2, temp_image_bright_gray, temp_image_bright, img_no, cnt3, cnt_left=None):
    if img_no == 1:
        mask = np.zeros(temp_image2.shape[:2], dtype="uint8")
        cv2.drawContours(mask, [cnt3], -1, 255, thickness=cv2.FILLED)
        masked_values = temp_image2[np.where(mask == 255)]
        max_value = np.max(masked_values)
        #rounded_max_value = round(max_value, 2)
        rounded_max_value = math.floor(max_value * 10) / 10
    else:
        mask = np.zeros(temp_image2.shape[:2], dtype="uint8")
        cv2.drawContours(mask, [cnt3], -1, 255, thickness=cv2.FILLED)


    # Initialize counters for each color
    white_count = 0
    yellow_count = 0
    orange_count = 0
    red_count = 0
    blue_count = 0
    # Total number of pixels in the contour
    total_count = np.sum(mask == 255)  


    # Apply color mapping logic directly to the original image
    for y in range(temp_image_bright_gray.shape[0]):
        for x in range(temp_image_bright_gray.shape[1]):
            # Only process pixels inside the contour
            if mask[y, x]:  
                pixel_value = temp_image2[y, x]
                if pixel_value >= max_value + 0.5:
                    temp_image_bright[y, x] = (255, 255, 255)  # White
                    white_count += 1
                elif max_value - 0.5 <= pixel_value < max_value + 0.5:
                    temp_image_bright[y, x] = (255, 255, 0)  # Yellow
                    yellow_count += 1
                elif max_value - 1.5 <= pixel_value < max_value - 0.5:
                    temp_image_bright[y, x] = (255, 165, 0)  # Orange
                    orange_count += 1
                elif max_value - 2.5 <= pixel_value < max_value - 1.5:
                    temp_image_bright[y, x] = (255, 0, 0)  # Red
                    red_count += 1
                else:
                    temp_image_bright[y, x] = (0, 0, 255)  # Blue
                    blue_count += 1

    
    # Compute ratios
    white_ratio = (white_count / total_count) * 100
    yellow_ratio = (yellow_count / total_count) * 100
    orange_ratio = (orange_count / total_count) * 100
    red_ratio = (red_count / total_count) * 100
    blue_ratio = (blue_count / total_count) * 100


    # Append ratios to corresponding lists
    if img_no != 1 and abs(white_ratio - white_ratios[-1]) >= 3: 
        white_ratios.append(white_ratios[-1])
    else:
        white_ratios.append(white_ratio)
    if img_no != 1 and abs(yellow_ratio - yellow_ratios[-1]) >= 3: 
        yellow_ratios.append(yellow_ratios[-1])
    else:
        yellow_ratios.append(yellow_ratio)
    if img_no != 1 and abs(orange_ratio - orange_ratios[-1]) >= 3: 
        orange_ratios.append(orange_ratios[-1])
    else:
        orange_ratios.append(orange_ratio)
    if img_no != 1 and abs(red_ratio - red_ratios[-1]) >= 5: 
        red_ratios.append(red_ratios[-1])
    else:
        red_ratios.append(red_ratio)
    if img_no != 1 and abs(blue_ratio - blue_ratios[-1]) >= 5: 
        blue_ratios.append(blue_ratios[-1])
    else:
        blue_ratios.append(blue_ratio)
    min_length = min(len(x_axis), len(white_ratios))
    x_axis = x_axis[:min_length]
    white_ratios = white_ratios[:min_length]
    yellow_ratios = yellow_ratios[:min_length]
    orange_ratios = orange_ratios[:min_length]
    red_ratios = red_ratios[:min_length]
    blue_ratios = blue_ratios[:min_length]


    # Add color mapped angiosomes to the unprocessed foot
    if cnt_left != None:
        cv2.drawContours(mask, [cnt_left], -1, 255, thickness=cv2.FILLED)
        # Apply color mapping logic directly to the original image
        for y in range(temp_image_bright_gray.shape[0]):
            for x in range(temp_image_bright_gray.shape[1]):
                if mask[y, x]:  # Only process pixels inside the contour
                    pixel_value = temp_image2[y, x]
                    if pixel_value >= max_value + 0.5:
                        temp_image_bright[y, x] = (255, 255, 255)  # White
                    elif max_value - 0.5 <= pixel_value < max_value + 0.5:
                        temp_image_bright[y, x] = (255, 255, 0)  # Yellow
                    elif max_value - 1.5 <= pixel_value < max_value - 0.5:
                        temp_image_bright[y, x] = (255, 165, 0)  # Orange
                    elif max_value - 2.5 <= pixel_value < max_value - 1.5:
                        temp_image_bright[y, x] = (255, 0, 0)  # Red
                    else:
                        temp_image_bright[y, x] = (0, 0, 255)  # Blue
    return temp_image_bright