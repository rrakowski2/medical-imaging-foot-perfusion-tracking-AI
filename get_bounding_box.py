#!/usr/bin/python

'''
Script/function to fetch contours of feet's masks and produce min-area bounding boxes - for the foot-tracking app

For presentation purposes, some sub-functions are embedded here rather than separated in a modular manner.
by Rafal May 2025
''' 


# Set environment
import cv2
import csv
import math
import numpy as np
from datetime import datetime
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import separate_left_right


# Initialize assigning None to all variables accessed/memorized globally
(
    process_frame, cx_arrow, old_cy_1, old_cy_2, old_cx_1, old_cx_2, corr,
    old_length_1, old_length_2, max_pixel_value, old_max_pixel_value,
    first_angle1, first_angle2, cnt3, cnt_left, x_axis, pass2,
    old_miniature_mask
) = (None,) * 18


# Assign False to specific flags
go_1 = go_2 = False


# Ensure the separate_left_right.cnt2 property is initialized 
separate_left_right.cnt2 = None


# Initialize data structure to store runtime-metadata
data = [[] for _ in range(5)]


# Function to create a descending sorted list of contour areas
def contour_area(contours):
    # create an empty list
    cnt_area = []
    # loop through all the contours
    for i in range(0,len(contours),1):
        # for each contour, use OpenCV to calculate the area of the contour
        cnt_area.append(cv2.contourArea(contours[i]))
    # Sort our list of contour areas in descending order
    list.sort(cnt_area, reverse=True)
    return cnt_area


# Main function to get feet's bounding boxes generate angiosome areas and populate them with a color coded temp isotherms
def get_bounding_box(contours, image, temp_image, temp_image2, img_no, csv_file, number_of_boxes=2): 
    global old_cy_1, old_cy_2, old_cx_1, old_cx_2, old_length_1, old_length_2, corr, max_pixel_value, old_max_pixel_value, data, first_angle1, first_angle2, cnt3, cnt_left, x_axis, go_1, go_2, pass2, cx_arrow, old_miniature_mask, process_frame 
    # Call our function to get the list of contour areas
    cnt_area = contour_area(contours)
    # Loop through each contour of our image
    count = 1
    cnt_1 = cnt_2 = cx_1 = cy_1 = cx_2 = cy_2 = None
    cnt_1_area = 0
    cnt_2_area = 0
    

    # Pick the feet's contours with their position
    for i in range(0,len(contours),1):
        cnt = contours[i]
        # Only draw the the largest two boxes
        if len(contours) < 2 or (cv2.contourArea(cnt) >= cnt_area[number_of_boxes -1] and cv2.contourArea(cnt) > 500):    # !!!general check
            cx, cy = 0, 0
            # Use OpenCV boundingRect function to get the details of the contour
            x,y,w,h = cv2.boundingRect(cnt)
            M = cv2.moments(cnt)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
            print("idx:", i, "| area:", cv2.contourArea(cnt), "| cx:", cx, "| cy:", cy, '| bbox', x, y, w, h)
            if count == 1:
                cnt_1 = cnt
                cx_1 = cx
                cy_1 = cy
            else:
                cnt_2 = cnt
                cx_2 = cx
                cy_2 = cy
            count += 1


    if cnt_1 is None:        
        #temp_evolution =  np.zeros_like(cv2.cvtColor(image, cv2.COLOR_GRAY2BGR))  
        process_frame = False 
        with open(csv_file, mode='a', newline='') as file2:
            writer = csv.writer(file2)
            # Append selected frame number to CSV file
            writer.writerow([img_no])       
        return np.zeros_like(image)
    

    # Differentiate feet's sidedeness
    image, pass1, pass2 = separate_left_right.tell_sidedeness(image, img_no, csv_file)
    if np.all(image == 0):
        result = image
        return result
        

    # In the case when right foot is not detected
    if pass2 == False:     
        if img_no != 1:                              
            #temp_evolution =  np.zeros_like(cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)) 
            process_frame = False
        with open(csv_file, mode='a', newline='') as file2:
            writer = csv.writer(file2)
            # Append selected frame number to CSV file
            writer.writerow([img_no])
        return np.zeros_like(image)   


    # In the case when left foot is detected
    if pass1 == True:
        # Draw the bounding box
        cv2.drawContours(image, separate_left_right.cnt1,0,(255,255,255),1,lineType=cv2.LINE_AA)
        #cv2.rectangle(image,(x,y),(x+w,y+h),(0,0,0),2)
        cv2.circle(image, (separate_left_right.cx1, separate_left_right.cy1), 1, (0, 0, 0), -1)
        rect1 = cv2.minAreaRect(separate_left_right.cnt1)
        if abs(rect1[2]) == 0.0:                         
            rect1 = ((rect1[0][0],  rect1[0][1]), (rect1[1][1], rect1[1][0]), 90)
        if old_cy_1 == 0 or go_1 == False:                                                           
            first_angle1 = rect1[2]
        #if abs(rect1[2] - first_angle1) > 50:
        #    rect1_temp = ((rect1[0][0],  rect1[0][1]), (rect1[1][1], rect1[1][0]), abs(90 - rect1[2]))
        #    rect1 = rect1_temp
        if abs(rect1[2]) < 50:
            rect1 = ((rect1[0][0],  rect1[0][1]), (rect1[1][1], rect1[1][0]), abs(90 - rect1[2]))
        #print('rotated box1', rect1)
        box1 = cv2.boxPoints(rect1)
        box1 = np.intp(box1) 


    # In the case when right foot is detected
    if pass2 == True:
        cx_arrow = separate_left_right.cx2           
        cv2.drawContours(image, separate_left_right.cnt2,0,(255,255,255),1,lineType=cv2.LINE_AA)
        #cv2.rectangle(image,(x,y),(x+w,y+h),(0,0,0),2)
        cv2.circle(image, (separate_left_right.cx2, separate_left_right.cy2), 1, (0, 0, 0), -1)
        

        # Scale the Bigmask
        def big_mask(contour, scale):                                  
            contour_scaled = contour - [separate_left_right.cx2, separate_left_right.cy2]
            contour_scaled = contour_scaled * scale
            contour_scaled = contour_scaled + [separate_left_right.cx2, separate_left_right.cy2]
            return contour_scaled.astype(np.int32)


        # Scale all contours
        rect2_original = cv2.minAreaRect(separate_left_right.cnt2) 
        cnt2_big = big_mask(separate_left_right.cnt2, 1.3) 
        rect2 = cv2.minAreaRect(cnt2_big)


        # Condition angular orientation of bounding box
        if abs(rect2[2]) == 0.0:
            rect2 = ((rect2[0][0],  rect2[0][1]), (rect2[1][1], rect2[1][0]), 90)     
        if old_cy_2 == 0 or go_2 == False:                                                
            first_angle2 = rect2[2]                                
        #if abs(rect2[2] - first_angle2) > 50:
        #    rect2_temp = ((rect2[0][0],  rect2[0][1]), (rect2[1][1], rect2[1][0]), abs(90 - rect2[2]))
        #    rect2 = rect2_temp         
        if abs(rect2[2]) < 50:
            rect2 = ((rect2[0][0],  rect2[0][1]), (rect2[1][1], rect2[1][0]), abs(90 - rect2[2]))                                          
        box2 = cv2.boxPoints(rect2)
        box2 = np.intp(box2)  


    # Process left foot beyond the first frame
    if  old_cy_1 != 0 and pass1 == True: 
        corr = False
        l_1 = max(rect1[1][0], rect1[1][1])
        delta_l_1 = l_1 - old_length_1
        side = 'R'


        # Cardinal points
        r = 0.85 * rect1[1][1] / 2
        r2 = 0.35 * rect1[1][1] 
        ux = separate_left_right.cx1 + math.sin(math.radians(rect1[2])) * r2
        uy = separate_left_right.cy1 - math.cos(math.radians(rect1[2])) * r2
        dx = separate_left_right.cx1 - math.sin(math.radians(rect1[2])) * r
        dy = separate_left_right.cy1 + math.cos(math.radians(rect1[2])) * r
        ldx = separate_left_right.cx1 - math.sin(math.radians(rect1[2])) * r / 0.85
        ldy = separate_left_right.cy1 + math.cos(math.radians(rect1[2])) * r / 0.85
        urx = ux + math.cos(math.radians(rect1[2])) * (3/8) * rect1[1][0]
        ury = uy + math.sin(math.radians(rect1[2])) * (3/8) * rect1[1][0]
        ulx = ux - math.cos(math.radians(rect1[2])) * (3/8) * rect1[1][0]
        uly = uy - math.sin(math.radians(rect1[2])) * (3/8) * rect1[1][0]
        drx = dx + math.cos(math.radians(rect1[2])) * (1/4) * rect1[1][0]
        dry = dy + math.sin(math.radians(rect1[2])) * (1/4) * rect1[1][0] 
        drux = drx + math.sin(math.radians(rect1[2])) * (1/4) * rect1[1][0] 
        druy = dry - math.cos(math.radians(rect1[2])) * (1/4) * rect1[1][0]
        dlx = dx - math.cos(math.radians(rect1[2])) * (1/4) * rect1[1][0]
        dly = dy - math.sin(math.radians(rect1[2])) * (1/4) * rect1[1][0]
        dlux = dlx + math.sin(math.radians(rect1[2])) * (1/4) * rect1[1][0] 
        dluy = dly - math.cos(math.radians(rect1[2])) * (1/4) * rect1[1][0]


        # Get the bottom point right
        def line_intersection(line1, line2):
            x1, y1, x2, y2 = line1
            x3, y3, x4, y4 = line2
            denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
            if denom == 0:
                return None  # Lines are parallel
            intersect_x = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / denom
            intersect_y = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / denom
            # Check if the intersection point is within the line segments
            if (min(x1, x2) <= intersect_x <= max(x1, x2) and min(y1, y2) <= intersect_y <= max(y1, y2) and
                min(x3, x4) <= intersect_x <= max(x3, x4) and min(y3, y4) <= intersect_y <= max(y3, y4)):
                return (int(intersect_x), int(intersect_y))
            else:
                return None


        cnt_poly = cv2.approxPolyDP(separate_left_right.cnt1, 0.01 * cv2.arcLength(separate_left_right.cnt1, True), True)
        intersection_points = []
        # Loop through each side of the bounding box
        for i in range(len(box1)):
            pt1 = box1[i]
            pt2 = box1[(i + 1) % len(box1)]
            line_box = (pt1[0], pt1[1], pt2[0], pt2[1])
            # Loop through each side of the contour
            for j in range(len(cnt_poly) - 1):
                pt3 = cnt_poly[j][0]
                pt4 = cnt_poly[j + 1][0]
                line_contour = (pt3[0], pt3[1], pt4[0], pt4[1])
                intersection = line_intersection(line_box, line_contour)
                if intersection:
                    intersection_points.append(intersection)


        intersection_points = np.unique(intersection_points, axis=0)
        #Draw the unique intersection points on the image
        #for point in intersection_points:
        #    cv2.circle(image, (point[0], point[1]), 5, (0, 0, 0), 2)
        print('inter points12', intersection_points)
        if intersection_points.size == 0 or list(max(intersection_points, key=lambda point: point[1]))[1] < ldy or corr == True: # intersection_points == []:
            intersection_points = [(int(ldx), int(ldy))]
        # Find the point with the max y-coordinate - one at the bottom
        bottom_point = list(max(intersection_points, key=lambda point: point[1]))
        bottom_point[0] = int(bottom_point[0])
        bottom_point[1] = int(bottom_point[1])
        print('bottom point12', bottom_point)


        # Get the side points right
        def extend_line(p1, p2, length=1000):
            """Extend the line in both directions by a given length."""
            direction = np.subtract(p2, p1)
            norm_direction = direction / np.linalg.norm(direction)
            extended_p1 = np.subtract(p1, norm_direction * length)
            extended_p2 = np.add(p2, norm_direction * length)
            return extended_p1, extended_p2
        

        def line_intersection1(p1, p2, q1, q2):
            """Check for intersection and return the intersection point if it exists"""
            def ccw(A, B, C):
                return (C[1]-A[1]) * (B[0]-A[0]) > (B[1]-A[1]) * (C[0]-A[0])
            if ccw(p1, q1, q2) != ccw(p2, q1, q2) and ccw(p1, p2, q1) != ccw(p1, p2, q2):
                # Calculate the intersection point
                A1, B1 = p2[1] - p1[1], p1[0] - p2[0]
                C1 = A1 * p1[0] + B1 * p1[1]
                A2, B2 = q2[1] - q1[1], q1[0] - q2[0]
                C2 = A2 * q1[0] + B2 * q1[1]
                determinant = A1 * B2 - A2 * B1
                if determinant != 0:
                    x = (B2 * C1 - B1 * C2) / determinant
                    y = (A1 * C2 - A2 * C1) / determinant
                    return [x, y]
            return None
        

        def find_intersections(contour_points, line_points):
            p1, p2 = line_points
            extended_p1, extended_p2 = extend_line(p1, p2)
            #print(f"Extended line points: {extended_p1}, {extended_p2}")
            intersections = []
            for i in range(len(contour_points)):
                q1, q2 = contour_points[i], contour_points[(i + 1) % len(contour_points)]
                #print(f"Checking segment: {q1}, {q2}")
                intersection = line_intersection1(extended_p1, extended_p2, q1, q2)
                if intersection is not None:
                    intersections.append(intersection)
                    print(f"Found intersection: {intersection}")
            return intersections
        

        # Define (by two points) the long straight line dividing angiosomes 
        point1 = (int(dlux), int(dluy))
        point2 = (int(drux), int(druy))
        line = [point1, point2]
        cnt3 = [tuple(point[0]) for point in separate_left_right.cnt1]
        #print('here',separate_left_right.cnt2)
        intersections = find_intersections(cnt3, line)
        print('here12', len(intersections))
        # Draw the intersection points
        if len(intersections) > 2:
            minimum = min(intersections, key=lambda p: p[0])
            maxaximum = max(intersections, key=lambda p: p[0])
            intersections = [minimum, maxaximum]
        print('here12', len(intersections))


        def select_point_with_smaller_x(point1, point2):
            if point1[0] < point2[0]:
                print('point12', point1, point1[0])
                point_left = point1
                point_right = point2
                return point_left, point_right
            else:
                point_left = point2
                point_right = point1
                return point_left, point_right
            

        if len(intersections) == 0:
            point_left, point_right =  [int(dlux), int(dluy)], [int(drux), int(druy)]
        else:
            point_left, point_right = select_point_with_smaller_x(intersections[0], intersections[1])
        

        #  Compute angiosomes' vertices
        ip = 0.85/2 * rect1[1][1] 
        px = separate_left_right.cx1 + math.sin(math.radians(rect1[2])) * ip - math.cos(math.radians(rect1[2])) * (1/4) * rect1[1][0] 
        py = separate_left_right.cy1 - math.cos(math.radians(rect1[2])) * ip - math.sin(math.radians(rect1[2])) * (1/4) * rect1[1][0] 


        # Define the vertices of angiosomes
        vertices1 = np.array([[int(ulx), int(uly)], point_left, [int(dx), int(dy)], [int(ux), int(uy)]], np.int32)
        vertices2 = np.array([[int(ux), int(uy)], [int(urx), int(ury)], point_right, [int(dx), int(dy)]], np.int32)
        vertices3 = np.array([point_left, [int(dx), int(dy)], bottom_point], np.int32)
        vertices4 = np.array([[int(dx), int(dy)], point_right,  bottom_point], np.int32)
        vertices1 = vertices1.reshape((-1, 1, 2))
        vertices2 = vertices2.reshape((-1, 1, 2))
        vertices3 = vertices3.reshape((-1, 1, 2))
        vertices4 = vertices4.reshape((-1, 1, 2))


        # Define the color (BGR) and thickness
        color = (0, 0, 0) 
        thickness = 1
        # Draw the triangle on the image
        cv2.polylines(image, [vertices1], isClosed=True, color=color, thickness=thickness)
        cv2.polylines(image, [vertices2], isClosed=True, color=color, thickness=thickness)
        cv2.polylines(image, [vertices3], isClosed=True, color=color, thickness=thickness)
        cv2.polylines(image, [vertices4], isClosed=True, color=color, thickness=thickness)


        # Define the center, axes, angle, start angle, and end angle of the ellipse
        center_coordinates = (int(px), int(py))
        axes_length = (int(1/8 * rect1[1][0]), int( 0.15/2 * rect1[1][1]))
        angle = rect1[2]
        start_angle = 0
        end_angle = 360
        # Define the color (BGR) and thickness
        color = (0, 0, 0) # Red color
        thickness = 1


        # Approximate the ellipse as a set of vertices
        num_vertices = 100
        ellipse_vertices = []
        for i in range(num_vertices):
            theta = np.deg2rad(start_angle + (i / num_vertices) * (end_angle - start_angle))
            x = int(center_coordinates[0] + axes_length[0] * np.cos(theta) * np.cos(np.deg2rad(angle)) - axes_length[1] * np.sin(theta) * np.sin(np.deg2rad(angle)))
            y = int(center_coordinates[1] + axes_length[0] * np.cos(theta) * np.sin(np.deg2rad(angle)) + axes_length[1] * np.sin(theta) * np.cos(np.deg2rad(angle)))
            ellipse_vertices.append((x, y))
        vertices5 = np.array(ellipse_vertices, dtype=np.int32)


    # Process left foot's first frame
    elif  old_cy_1 == 0 and pass1 == True:#  and count <= 2:
        print('cnt', type(separate_left_right.cnt1))
        old_length_1 = max(rect1[1][0], rect1[1][1])
        #rect_corr = rect
        side = 'R'


        # Cardinal points
        r = 0.85 * rect1[1][1] / 2
        r2 = 0.35 * rect1[1][1] 
        ux = separate_left_right.cx1 + math.sin(math.radians(rect1[2])) * r2
        uy = separate_left_right.cy1 - math.cos(math.radians(rect1[2])) * r2
        dx = separate_left_right.cx1 - math.sin(math.radians(rect1[2])) * r
        dy = separate_left_right.cy1 + math.cos(math.radians(rect1[2])) * r
        ldx = separate_left_right.cx1 - math.sin(math.radians(rect1[2])) * r / 0.85
        ldy = separate_left_right.cy1 + math.cos(math.radians(rect1[2])) * r / 0.85
        urx = ux + math.cos(math.radians(rect1[2])) * (3/8) * rect1[1][0]
        ury = uy + math.sin(math.radians(rect1[2])) * (3/8) * rect1[1][0]
        ulx = ux - math.cos(math.radians(rect1[2])) * (3/8) * rect1[1][0]
        uly = uy - math.sin(math.radians(rect1[2])) * (3/8) * rect1[1][0]
        drx = dx + math.cos(math.radians(rect1[2])) * (1/4) * rect1[1][0]
        dry = dy + math.sin(math.radians(rect1[2])) * (1/4) * rect1[1][0] 
        drux = drx + math.sin(math.radians(rect1[2])) * (1/4) * rect1[1][0] 
        druy = dry - math.cos(math.radians(rect1[2])) * (1/4) * rect1[1][0]
        dlx = dx - math.cos(math.radians(rect1[2])) * (1/4) * rect1[1][0]
        dly = dy - math.sin(math.radians(rect1[2])) * (1/4) * rect1[1][0]
        dlux = dlx + math.sin(math.radians(rect1[2])) * (1/4) * rect1[1][0] 
        dluy = dly - math.cos(math.radians(rect1[2])) * (1/4) * rect1[1][0]


        # Get to bottom point right
        def line_intersection(line1, line2):
            x1, y1, x2, y2 = line1
            x3, y3, x4, y4 = line2
            denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
            if denom == 0:
                return None  # Lines are parallel
            intersect_x = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / denom
            intersect_y = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / denom
            # Check if the intersection point is within the line segments
            if (min(x1, x2) <= intersect_x <= max(x1, x2) and min(y1, y2) <= intersect_y <= max(y1, y2) and
                min(x3, x4) <= intersect_x <= max(x3, x4) and min(y3, y4) <= intersect_y <= max(y3, y4)):
                return (int(intersect_x), int(intersect_y))
            else:
                return None
            

        cnt_poly = cv2.approxPolyDP(separate_left_right.cnt1, 0.01 * cv2.arcLength(separate_left_right.cnt1, True), True)
        intersection_points = []
        # Loop through each side of the bounding box
        for i in range(len(box1)):
            pt1 = box1[i]
            pt2 = box1[(i + 1) % len(box1)]
            line_box = (pt1[0], pt1[1], pt2[0], pt2[1])
            # Loop through each side of the contour
            for j in range(len(cnt_poly) - 1):
                pt3 = cnt_poly[j][0]
                pt4 = cnt_poly[j + 1][0]
                line_contour = (pt3[0], pt3[1], pt4[0], pt4[1])
                intersection = line_intersection(line_box, line_contour)
                if intersection:
                    intersection_points.append(intersection)


        intersection_points = np.unique(intersection_points, axis=0)
        # Draw the unique intersection points on the image
        #for point in intersection_points:
        #    cv2.circle(image, (point[0], point[1]), 5, (0, 255, 0), -1)
        if intersection_points.size == 0 or list(max(intersection_points, key=lambda point: point[1]))[1] < ldy: # intersection_points == []:
           intersection_points = [((ldx), (ldy))]
                                  
        print('inter points11', intersection_points)
        # Find the point with the max y-coordinate - one at the bottom
        bottom_point = list(max(intersection_points, key=lambda point: point[1]))
        #bottom_point = list(cnt[cnt[:, :, 1].argmax()][0]) # max y (most bottom)
        bottom_point[0] = int(bottom_point[0])
        bottom_point[1] = int(bottom_point[1])
        print('bottom point11', (bottom_point))
         
        
        # Get the side points right
        def extend_line(p1, p2, length=1000):
            """Extend the line in both directions by a given length."""
            direction = np.subtract(p2, p1)
            norm_direction = direction / np.linalg.norm(direction)
            extended_p1 = np.subtract(p1, norm_direction * length)
            extended_p2 = np.add(p2, norm_direction * length)
            return extended_p1, extended_p2
        

        def line_intersection1(p1, p2, q1, q2):
            """Check for intersection and return the intersection point if it exists."""
            def ccw(A, B, C):
                return (C[1]-A[1]) * (B[0]-A[0]) > (B[1]-A[1]) * (C[0]-A[0])
            if ccw(p1, q1, q2) != ccw(p2, q1, q2) and ccw(p1, p2, q1) != ccw(p1, p2, q2):
                # Calculate the intersection point
                A1, B1 = p2[1] - p1[1], p1[0] - p2[0]
                C1 = A1 * p1[0] + B1 * p1[1]
                A2, B2 = q2[1] - q1[1], q1[0] - q2[0]
                C2 = A2 * q1[0] + B2 * q1[1]
                determinant = A1 * B2 - A2 * B1
                if determinant != 0:
                    x = (B2 * C1 - B1 * C2) / determinant
                    y = (A1 * C2 - A2 * C1) / determinant
                    return [x, y]
            return None
        

        def find_intersections(contour_points, line_points):
            p1, p2 = line_points
            extended_p1, extended_p2 = extend_line(p1, p2)
            #print(f"Extended line points: {extended_p1}, {extended_p2}")
            intersections = []
            for i in range(len(contour_points)):
                q1, q2 = contour_points[i], contour_points[(i + 1) % len(contour_points)]
                #print(f"Checking segment: {q1}, {q2}")
                intersection = line_intersection1(extended_p1, extended_p2, q1, q2)
                if intersection is not None:
                    intersections.append(intersection)
                    print(f"Found intersection: {intersection}")
            return intersections
        

        # Define (by two points) the long straight line dividing angiosomes 
        point1 = (int(dlux), int(dluy))
        point2 = (int(drux), int(druy))
        line = [point1, point2]
        cnt3 = [tuple(point[0]) for point in separate_left_right.cnt1]
        #print('here',separate_left_right.cnt2)
        intersections = find_intersections(cnt3, line)
        print('here11', len(intersections))
        # Draw the intersection points
        if len(intersections) > 2:
            minimum = min(intersections, key=lambda p: p[0])
            maxaximum = max(intersections, key=lambda p: p[0])
            intersections = [minimum, maxaximum]


        def select_point_with_smaller_x(point1, point2):
            if point1[0] < point2[0]:
                point_left = point1
                point_right = point2
                return point_left, point_right
            else:
                point_left = point2
                point_right = point1
                return point_left, point_right
        if len(intersections) == 0:
            point_left, point_right =  [int(dlux), int(dluy)], [int(drux), int(druy)]
        else:
            point_left, point_right = select_point_with_smaller_x(intersections[0], intersections[1])
        

        #  Compute angiosomes' vertices
        ip = 0.85/2 * rect1[1][1] 
        px = separate_left_right.cx1 + math.sin(math.radians(rect1[2])) * ip - math.cos(math.radians(rect1[2])) * (1/4) * rect1[1][0] 
        py = separate_left_right.cy1 - math.cos(math.radians(rect1[2])) * ip - math.sin(math.radians(rect1[2])) * (1/4) * rect1[1][0] 


        # Define the vertices of angiosomes
        vertices1 = np.array([[int(ulx), int(uly)], point_left, [int(dx), int(dy)], [int(ux), int(uy)]], np.int32)
        vertices2 = np.array([[int(ux), int(uy)], [int(urx), int(ury)], point_right, [int(dx), int(dy)]], np.int32)
        vertices3 = np.array([point_left, [int(dx), int(dy)], bottom_point], np.int32)
        vertices4 = np.array([[int(dx), int(dy)], point_right,  bottom_point], np.int32)
        vertices1 = globals()[f'vertices1'] = vertices1.reshape((-1, 1, 2))
        vertices2 = globals()[f'vertices2'] = vertices2.reshape((-1, 1, 2))
        vertices3 = globals()[f'vertices3'] = vertices3.reshape((-1, 1, 2))
        vertices4 = globals()[f'vertices4'] = vertices4.reshape((-1, 1, 2))
        

        # Define the color (BGR) and thickness
        color = (0, 0, 0) 
        #color = (250, 250, 250) 
        thickness = 1
        # Draw the triangle on the image
        cv2.polylines(image, [vertices1], isClosed=True, color=color, thickness=thickness)
        cv2.polylines(image, [vertices2], isClosed=True, color=color, thickness=thickness)
        cv2.polylines(image, [vertices3], isClosed=True, color=color, thickness=thickness)
        cv2.polylines(image, [vertices4], isClosed=True, color=color, thickness=thickness)


        # Define the center, axes, angle, start angle, and end angle of the ellipse
        center_coordinates = (int(px), int(py))
        axes_length = (int(1/8 * rect1[1][0]), int( 0.15/2 * rect1[1][1]))
        angle = rect1[2]
        start_angle = 0
        end_angle = 360
        # Define the color (BGR) and thickness
        color = (0, 0, 0) # Red color
        thickness = 1


        # Approximate the ellipse as a set of vertices
        num_vertices = 100
        ellipse_vertices = []
        for i in range(num_vertices):
            theta = np.deg2rad(start_angle + (i / num_vertices) * (end_angle - start_angle))
            x = int(center_coordinates[0] + axes_length[0] * np.cos(theta) * np.cos(np.deg2rad(angle)) - axes_length[1] * np.sin(theta) * np.sin(np.deg2rad(angle)))
            y = int(center_coordinates[1] + axes_length[0] * np.cos(theta) * np.sin(np.deg2rad(angle)) + axes_length[1] * np.sin(theta) * np.cos(np.deg2rad(angle)))
            ellipse_vertices.append((x, y))
        vertices5 = np.array(ellipse_vertices, dtype=np.int32)


    # Process right foot beyond first frame
    if old_cy_2 != 0 and pass2 == True:
        side = 'L'
        l_2 = max(rect2[1][0], rect2[1][1])
        delta_l_2 = l_2 - old_length_2


        # Cardinal points
        l = 0.85 * rect2[1][0] / 2
        l2 = 0.35 * rect2[1][0] 
        ux = separate_left_right.cx2 - math.cos(math.radians(rect2[2])) * l2 *0.77         
        uy = separate_left_right.cy2 - math.sin(math.radians(rect2[2])) * l2 *0.77          
        dx = separate_left_right.cx2 + math.cos(math.radians(rect2[2])) * l  *0.8          
        dy = separate_left_right.cy2 + math.sin(math.radians(rect2[2])) * l  *0.8          
        ldx = separate_left_right.cx2 + math.cos(math.radians(rect2[2])) * l / 0.85
        ldy = separate_left_right.cy2 + math.sin(math.radians(rect2[2])) * l / 0.85
        urx = ux + math.sin(math.radians(rect2[2])) * (3/8) * rect2[1][1]
        ury = uy - math.cos(math.radians(rect2[2])) * (3/8) * rect2[1][1]
        ulx = ux - math.sin(math.radians(rect2[2])) * (3/8) * rect2[1][1]
        uly = uy + math.cos(math.radians(rect2[2])) * (3/8) * rect2[1][1]
        drx = dx + math.sin(math.radians(rect2[2])) * (1/4) * rect2[1][1]
        drux = drx - math.cos(math.radians(rect2[2])) * (1/4) * rect2[1][1] 
        dry = dy - math.cos(math.radians(rect2[2])) * (1/4) * rect2[1][1] 
        druy = dry - math.sin(math.radians(rect2[2])) * (1/4) * rect2[1][1]
        dlx = dx - math.sin(math.radians(rect2[2])) * (1/4) * rect2[1][1]
        dlux = dlx - math.cos(math.radians(rect2[2])) * (1/4) * rect2[1][1] 
        dly = dy + math.cos(math.radians(rect2[2])) * (1/4) * rect2[1][1]
        dluy = dly - math.sin(math.radians(rect2[2])) * (1/4) * rect2[1][1]

        
        # Get the bottom point right
        def line_intersection(line1, line2):
            x1, y1, x2, y2 = line1
            x3, y3, x4, y4 = line2
            denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
            if denom == 0:
                return None  # Lines are parallel
            intersect_x = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / denom
            intersect_y = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / denom
            # Check if the intersection point is within the line segments
            if (min(x1, x2) <= intersect_x <= max(x1, x2) and min(y1, y2) <= intersect_y <= max(y1, y2) and
                min(x3, x4) <= intersect_x <= max(x3, x4) and min(y3, y4) <= intersect_y <= max(y3, y4)):
                return (int(intersect_x), int(intersect_y))
            else:
                return None

        # Define excessive mask to get rounded angiosomes
        def big_mask(contour, scale):
            contour_scaled = contour - [separate_left_right.cx2, separate_left_right.cy2]
            contour_scaled = contour_scaled * scale
            contour_scaled = contour_scaled + [separate_left_right.cx2, separate_left_right.cy2]
            return contour_scaled.astype(np.int32)


        # Scale all contours
        cnt2_big = big_mask(separate_left_right.cnt2, 1.3)  #was 0.95
        cnt_poly = cv2.approxPolyDP(cnt2_big, 0.01 * cv2.arcLength(cnt2_big, True), True)   # bigmask
        intersection_points = []


        # Loop through each side of the bounding box
        for i in range(len(box2)):
            pt1 = box2[i]
            pt2 = box2[(i + 1) % len(box2)]
            line_box = (pt1[0], pt1[1], pt2[0], pt2[1])
            # Loop through each side of the contour
            for j in range(len(cnt_poly) - 1):
                pt3 = cnt_poly[j][0]
                pt4 = cnt_poly[j + 1][0]
                line_contour = (pt3[0], pt3[1], pt4[0], pt4[1])
                intersection = line_intersection(line_box, line_contour)
                if intersection:
                    intersection_points.append(intersection)
        intersection_points = np.unique(intersection_points, axis=0)
        print('inter points22', intersection_points)


        # Find the point with the max y-coordinate - one at the bottom  
        if len(intersection_points) > 0:                             
            bottom_point = list(max(intersection_points, key=lambda point: point[1]))
            if abs(ldx - bottom_point[0]) > 20 or abs(ldy - bottom_point[1]) > 20:
                bottom_point = [int(ldx), int(ldy)]
            #bottom_point[0] = int(bottom_point[0])
            #bottom_point[1] = int(bottom_point[1])
        else:
            bottom_point = [int(ldx), int(ldy)]
        print('bottom point22', bottom_point)


       # Get the down-side points right
        def extend_line(p1, p2, length=1000):
            """Extend the line in both directions by a given length."""
            direction = np.subtract(p2, p1)
            norm_direction = direction / np.linalg.norm(direction)
            extended_p1 = np.subtract(p1, norm_direction * length)
            extended_p2 = np.add(p2, norm_direction * length)
            return extended_p1, extended_p2
        

        def line_intersection1(p1, p2, q1, q2):
            """Check for intersection and return the intersection point if it exists."""
            def ccw(A, B, C):
                return (C[1]-A[1]) * (B[0]-A[0]) > (B[1]-A[1]) * (C[0]-A[0])
            if ccw(p1, q1, q2) != ccw(p2, q1, q2) and ccw(p1, p2, q1) != ccw(p1, p2, q2):
                # Calculate the intersection point
                A1, B1 = p2[1] - p1[1], p1[0] - p2[0]
                C1 = A1 * p1[0] + B1 * p1[1]
                A2, B2 = q2[1] - q1[1], q1[0] - q2[0]
                C2 = A2 * q1[0] + B2 * q1[1]
                determinant = A1 * B2 - A2 * B1
                if determinant != 0:
                    x = (B2 * C1 - B1 * C2) / determinant
                    y = (A1 * C2 - A2 * C1) / determinant
                    return [x, y]
            return None
        

        def find_intersections(contour_points, line_points):
            p1, p2 = line_points
            extended_p1, extended_p2 = extend_line(p1, p2)
            print(f"Extended line points22: {extended_p1}, {extended_p2}")
            intersections = []
            for i in range(len(contour_points)):
                q1, q2 = contour_points[i], contour_points[(i + 1) % len(contour_points)]
                #print(f"Checking segment: {q1}, {q2}")
                intersection = line_intersection1(extended_p1, extended_p2, q1, q2)
                if intersection is not None:
                    intersections.append(intersection)
                    print(f"Found intersection: {intersection}")
            return intersections
        

        # Define (by two points) the long straight line dividing angiosomes 
        point1 = (int(dlux), int(dluy))
        point2 = (int(drux), int(druy))
        line = [point1, point2]
        cnt4 = [tuple(point[0]) for point in cnt2_big]                    
        #print('here',separate_left_right.cnt2)
        intersections = find_intersections(cnt4, line)
        print('here22', len(intersections))
        # Draw the intersection points
        if len(intersections) > 2:
            minimum = min(intersections, key=lambda p: p[0])
            maxaximum = max(intersections, key=lambda p: p[0])
            intersections = [minimum, maxaximum]


        def select_point_with_smaller_x(point1, point2):
            if point1[0] < point2[0]:
                point_left = point1
                point_right = point2
                return point_left, point_right
            else:
                point_left = point2
                point_right = point1
                return point_left, point_right
        #print('inter', len(intersections), (intersections))
        if len(intersections) == 0:
            point_left, point_right =  [int(dlux), int(dluy)], [int(drux), int(druy)]
        else:
            point_left, point_right = select_point_with_smaller_x(intersections[0], intersections[1])

        # Ammend Rois' points and get point x point(ux, uy) & bottom_point
        vert_line = (int(ux), int(uy), bottom_point[0], bottom_point[1])
        # point_left & point_right
        horiz_line = (int(dlux), int(dluy), int(drux), int(druy))


        # Draw the line
        def line_intersection2(line1, line2):
            x1, y1, x2, y2 = line1
            x3, y3, x4, y4 = line2
            denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
            if denom == 0:
                return None  # Lines are parallel
            intersect_x = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / denom
            intersect_y = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / denom
            return (int(intersect_x), int(intersect_y))
        x_point = line_intersection2(vert_line, horiz_line)
        print('x_point', vert_line, horiz_line , x_point)

        
        #  Compute angiosomes' vertices
        ip = 0.85 * rect2[1][0] / 2 
        px = separate_left_right.cx2 - math.cos(math.radians(rect2[2])) * ip + math.sin(math.radians(rect2[2])) * (3/16) * rect2[1][1] 
        py = separate_left_right.cy2 - math.sin(math.radians(rect2[2])) * ip - math.cos(math.radians(rect2[2])) * (3/16) * rect2[1][1] 
        ip_original = 0.85 * rect2_original[1][0] /2 
        px_original = separate_left_right.cx2 - math.cos(math.radians(rect2_original[2])) * ip_original + math.sin(math.radians(rect2_original[2])) * (3/16) * rect2_original[1][1] 
        py_original = separate_left_right.cy2 - math.sin(math.radians(rect2_original[2])) * ip_original - math.cos(math.radians(rect2_original[2])) * (3/16) * rect2_original[1][1] 
        
        
        # Define the vertices of angiosomes
        try:
            vertices1 = np.array([[int(ulx), int(uly)], point_left, [x_point[0], x_point[1]], [int(ux), int(uy)]], np.int32) # replaced point x instead of dx,dy
            vertices2 = np.array([[int(ux), int(uy)], [int(urx), int(ury)], point_right, [x_point[0], x_point[1]]], np.int32)  # replaced point x instead of dx,dy
            vertices3 = np.array([point_left, [x_point[0], x_point[1]], bottom_point], np.int32)  # replaced point x instead of dx,dy  & 
            vertices4 = np.array([[x_point[0], x_point[1]], point_right,  bottom_point], np.int32)
        except Exception as e:
            result = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            with open(csv_file, mode='a', newline='') as file2:
                writer = csv.writer(file2)
                # Append selected frame number to CSV file
                writer.writerow([img_no])
            return np.zeros_like(result)
        

        vertices1 = vertices1.reshape((-1, 1, 2))
        vertices2 = vertices2.reshape((-1, 1, 2))
        vertices3 = vertices3.reshape((-1, 1, 2))
        vertices4 = vertices4.reshape((-1, 1, 2))


        # Define the color and thickness
        color = (0,0,0) 
        thickness = 1


        # Draw the triangle on the image
        cv2.polylines(image, [vertices1], isClosed=True, color=color, thickness=thickness)
        cv2.polylines(image, [vertices2], isClosed=True, color=color, thickness=thickness)
        cv2.polylines(image, [vertices3], isClosed=True, color=color, thickness=thickness)
        cv2.polylines(image, [vertices4], isClosed=True, color=color, thickness=thickness)
        
        
        # Define the center, axes, angle, start angle, and end angle of the ellipse
        center_coordinates = (int(px_original), int(py_original))
        axes_length = (int(1/8 * rect2_original[1][1]), int(0.15/2 * rect2_original[1][0]))
        angle = rect2_original[2] - 90
        start_angle = 0
        end_angle = 360
        # Define the color (BGR) and thickness
        color = (0,0,0) 
        thickness = 1
        

        # Approximate the ellipse as a set of vertices
        num_vertices = 100
        ellipse_vertices = []
        for i in range(num_vertices):
            theta = np.deg2rad(start_angle + (i / num_vertices) * (end_angle - start_angle))
            x = int(center_coordinates[0] + axes_length[0] * np.cos(theta) * np.cos(np.deg2rad(angle)) - axes_length[1] * np.sin(theta) * np.sin(np.deg2rad(angle)))
            y = int(center_coordinates[1] + axes_length[0] * np.cos(theta) * np.sin(np.deg2rad(angle)) + axes_length[1] * np.sin(theta) * np.cos(np.deg2rad(angle)))
            ellipse_vertices.append((x, y))
        vertices5 = np.array(ellipse_vertices, dtype=np.int32)
        

    # Process right foot's first frame
    elif old_cy_2 == 0 and pass2 == True:
        side = 'L'
        old_length_2 = max(rect2[1][0], rect2[1][1])


        # Cardinal points
        l = 0.85 * rect2[1][0] / 2
        l2 = 0.35 * rect2[1][0] 
        ux = separate_left_right.cx2 - math.cos(math.radians(rect2[2])) * l2 *0.77          
        uy = separate_left_right.cy2 - math.sin(math.radians(rect2[2])) * l2 *0.77           
        dx = separate_left_right.cx2 + math.cos(math.radians(rect2[2])) * l *0.8          
        dy = separate_left_right.cy2 + math.sin(math.radians(rect2[2])) * l *0.8          
        ldx = separate_left_right.cx2 + math.cos(math.radians(rect2[2])) * l / 0.85
        ldy = separate_left_right.cy2 + math.sin(math.radians(rect2[2])) * l / 0.85
        urx = ux + math.sin(math.radians(rect2[2])) * (3/8) * rect2[1][1]
        ury = uy - math.cos(math.radians(rect2[2])) * (3/8) * rect2[1][1]
        ulx = ux - math.sin(math.radians(rect2[2])) * (3/8) * rect2[1][1]
        uly = uy + math.cos(math.radians(rect2[2])) * (3/8) * rect2[1][1]
        drx = dx + math.sin(math.radians(rect2[2])) * (1/4) * rect2[1][1]
        drux = drx - math.cos(math.radians(rect2[2])) * (1/4) * rect2[1][1] 
        dry = dy - math.cos(math.radians(rect2[2])) * (1/4) * rect2[1][1] 
        druy = dry - math.sin(math.radians(rect2[2])) * (1/4) * rect2[1][1]
        dlx = dx - math.sin(math.radians(rect2[2])) * (1/4) * rect2[1][1]
        dlux = dlx - math.cos(math.radians(rect2[2])) * (1/4) * rect2[1][1] 
        dly = dy + math.cos(math.radians(rect2[2])) * (1/4) * rect2[1][1]
        dluy = dly - math.sin(math.radians(rect2[2])) * (1/4) * rect2[1][1]

              
        # Get the bottom point right
        def line_intersection(line1, line2):
            x1, y1, x2, y2 = line1
            x3, y3, x4, y4 = line2
            denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
            if denom == 0:
                return None  # Lines are parallel
            intersect_x = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / denom
            intersect_y = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / denom
            # Check if the intersection point is within the line segments
            if (min(x1, x2) <= intersect_x <= max(x1, x2) and min(y1, y2) <= intersect_y <= max(y1, y2) and
                min(x3, x4) <= intersect_x <= max(x3, x4) and min(y3, y4) <= intersect_y <= max(y3, y4)):
                return (int(intersect_x), int(intersect_y))
            else:
                return None
            

        def big_mask(contour, scale):
            contour_scaled = contour - [separate_left_right.cx2, separate_left_right.cy2]
            contour_scaled = contour_scaled * scale
            contour_scaled = contour_scaled + [separate_left_right.cx2, separate_left_right.cy2]
            return contour_scaled.astype(np.int32)


        # Scale all contours
        cnt2_big = big_mask(separate_left_right.cnt2, 1.3) 
        cnt_poly = cv2.approxPolyDP(cnt2_big, 0.01 * cv2.arcLength(cnt2_big, True), True)  
        intersection_points = []
        # Loop through each side of the bounding box
        for i in range(len(box2)):
            pt1 = box2[i]
            pt2 = box2[(i + 1) % len(box2)]
            line_box = (pt1[0], pt1[1], pt2[0], pt2[1])
            # Loop through each side of the contour
            for j in range(len(cnt_poly) - 1):
                pt3 = cnt_poly[j][0]
                pt4 = cnt_poly[j + 1][0]
                line_contour = (pt3[0], pt3[1], pt4[0], pt4[1])
                intersection = line_intersection(line_box, line_contour)
                if intersection:
                    intersection_points.append(intersection)


        intersection_points = np.unique(intersection_points, axis=0)
        print('inter points21', intersection_points)
        if intersection_points.size == 0 or list(max(intersection_points, key=lambda point: point[1]))[1] < ldy: # intersection_points == []:
            intersection_points = [(int(ldx), int(ldy))]

        # Find the point with the max y-coordinate - the one at the bottom
        bottom_point = list(max(intersection_points, key=lambda point: point[1]))
        bottom_point[0] = int(bottom_point[0])
        bottom_point[1] = int(bottom_point[1])
        print('bottom point21', bottom_point)


        # Get the side points right
        def extend_line(p1, p2, length=1000):
            """Extend the line in both directions by a given length."""
            direction = np.subtract(p2, p1)
            norm_direction = direction / np.linalg.norm(direction)
            extended_p1 = np.subtract(p1, norm_direction * length)
            extended_p2 = np.add(p2, norm_direction * length)
            return extended_p1, extended_p2
        

        def line_intersection1(p1, p2, q1, q2):
            """Check for intersection and return the intersection point if it exists."""
            def ccw(A, B, C):
                return (C[1]-A[1]) * (B[0]-A[0]) > (B[1]-A[1]) * (C[0]-A[0])
            if ccw(p1, q1, q2) != ccw(p2, q1, q2) and ccw(p1, p2, q1) != ccw(p1, p2, q2):
                # Calculate the intersection point
                A1, B1 = p2[1] - p1[1], p1[0] - p2[0]
                C1 = A1 * p1[0] + B1 * p1[1]
                A2, B2 = q2[1] - q1[1], q1[0] - q2[0]
                C2 = A2 * q1[0] + B2 * q1[1]
                determinant = A1 * B2 - A2 * B1
                if determinant != 0:
                    x = (B2 * C1 - B1 * C2) / determinant
                    y = (A1 * C2 - A2 * C1) / determinant
                    return [x, y]
            return None
        

        def find_intersections(contour_points, line_points):
            p1, p2 = line_points
            extended_p1, extended_p2 = extend_line(p1, p2)
            print(f"Extended line points21: {extended_p1}, {extended_p2}")
            intersections = []
            for i in range(len(contour_points)):
                q1, q2 = contour_points[i], contour_points[(i + 1) % len(contour_points)]
                #print(f"Checking segment: {q1}, {q2}")
                intersection = line_intersection1(extended_p1, extended_p2, q1, q2)
                if intersection is not None:
                    intersections.append(intersection)
                    print(f"Found intersection: {intersection}")
            return intersections
        

        # Define (by two points) the long straight line dividing angiosomes 
        point1 = (int(dlux), int(dluy))
        point2 = (int(drux), int(druy))
        line = [point1, point2]
        cnt3 = [tuple(point[0]) for point in cnt2_big]                    
        #print('here',separate_left_right.cnt2)
        intersections = find_intersections(cnt3, line)
        if len(intersections) > 2:
            minimum = min(intersections, key=lambda p: p[0])
            maxaximum = max(intersections, key=lambda p: p[0])
            intersections = [minimum, maxaximum]
        print('here21', len(intersections))
        

        def select_point_with_smaller_x(point1, point2):
            if point1[0] < point2[0]:
                point_left = point1
                point_right = point2
                return point_left, point_right
            else:
                point_left = point2
                point_right = point1
                return point_left, point_right
    
        if len(intersections) == 0:
            point_left, point_right =  [int(dlux), int(dluy)], [int(drux), int(druy)]
        else:
            point_left, point_right = select_point_with_smaller_x(intersections[0], intersections[1])

        # Ammend Rois points and get point x point(ux, uy) & bottom_point
        vert_line = (int(ux), int(uy), bottom_point[0], bottom_point[1])
        # point_left & point_right
        horiz_line = (int(dlux), int(dluy), int(drux), int(druy))


        # Draw the line
        def line_intersection2(line1, line2):
            x1, y1, x2, y2 = line1
            x3, y3, x4, y4 = line2
            denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
            if denom == 0:
                return None  # Lines are parallel
            intersect_x = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / denom
            intersect_y = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / denom
            return (int(intersect_x), int(intersect_y))
        x_point = line_intersection2(vert_line, horiz_line)
        print('x_point', vert_line, horiz_line , x_point)


        #  Compute angiosomes' vertices
        ip = 0.85 * rect2[1][0] /2 
        px = separate_left_right.cx2 - math.cos(math.radians(rect2[2])) * ip + math.sin(math.radians(rect2[2])) * (3/16) * rect2[1][1] 
        py = separate_left_right.cy2 - math.sin(math.radians(rect2[2])) * ip - math.cos(math.radians(rect2[2])) * (3/16) * rect2[1][1] 
        ip_original = 0.85 * rect2_original[1][0] /2 
        px_original = separate_left_right.cx2 - math.cos(math.radians(rect2_original[2])) * ip_original + math.sin(math.radians(rect2_original[2])) * (3/16) * rect2_original[1][1] 
        py_original = separate_left_right.cy2 - math.sin(math.radians(rect2_original[2])) * ip_original - math.cos(math.radians(rect2_original[2])) * (3/16) * rect2_original[1][1] 
        
        
        # Define the vertices of angiosomes
        try:
            vertices1 = np.array([[int(ulx), int(uly)], point_left, [x_point[0], x_point[1]], [int(ux), int(uy)]], np.int32) # replaced point x instead of dx,dy
            vertices2 = np.array([[int(ux), int(uy)], [int(urx), int(ury)], point_right, [x_point[0], x_point[1]]], np.int32)  # replaced point x instead of dx,dy
            vertices3 = np.array([point_left, [x_point[0], x_point[1]], bottom_point], np.int32)  # replaced point x instead of dx,dy  & 
            vertices4 = np.array([[x_point[0], x_point[1]], point_right,  bottom_point], np.int32)
        except Exception as e:
            result = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            with open(csv_file, mode='a', newline='') as file2:
                writer = csv.writer(file2)
                # Append selected frame number to CSV file
                writer.writerow([img_no])
            return np.zeros_like(result)
        

        vertices1 = vertices1.reshape((-1, 1, 2))
        vertices2 = vertices2.reshape((-1, 1, 2))
        vertices3 = vertices3.reshape((-1, 1, 2))
        vertices4 = vertices4.reshape((-1, 1, 2))

        # Define the color (BGR) and thickness
        color = (0, 0, 0) 
        thickness = 1

        # Draw the triangle on the image
        cv2.polylines(image, [vertices1], isClosed=True, color=color, thickness=thickness)
        cv2.polylines(image, [vertices2], isClosed=True, color=color, thickness=thickness)
        cv2.polylines(image, [vertices3], isClosed=True, color=color, thickness=thickness)
        cv2.polylines(image, [vertices4], isClosed=True, color=color, thickness=thickness)

        # Define the center, axes, angle, start angle, and end angle of the ellipse
        center_coordinates = (int(px_original), int(py_original))
        axes_length = (int(1/8 * rect2_original[1][1]), int(0.15/2 * rect2_original[1][0]))
        angle = rect2_original[2] - 90
        start_angle = 0
        end_angle = 360
        # Define the color (BGR) and thickness
        color = (0,0,0) # Red color
        thickness = 1


        # Approximate the ellipse as a set of vertices
        num_vertices = 100
        ellipse_vertices = []
        for i in range(num_vertices):
            theta = np.deg2rad(start_angle + (i / num_vertices) * (end_angle - start_angle))
            x = int(center_coordinates[0] + axes_length[0] * np.cos(theta) * np.cos(np.deg2rad(angle)) - axes_length[1] * np.sin(theta) * np.sin(np.deg2rad(angle)))
            y = int(center_coordinates[1] + axes_length[0] * np.cos(theta) * np.sin(np.deg2rad(angle)) + axes_length[1] * np.sin(theta) * np.cos(np.deg2rad(angle)))
            ellipse_vertices.append((x, y))
        vertices5 = np.array(ellipse_vertices, dtype=np.int32)

    
    # Get bounding box for left foot
    if pass1 == True:
        print('rotated box1', rect1)
        box1 = cv2.boxPoints(rect1)
        #print('box points', box)
        box1 = np.intp(box1)
        cv2.drawContours(image,[box1],0,(255,0,0),1)


    # Get bounding box for right foot
    if pass2 == True:
        print('rotated box2', rect2)
        box2 = cv2.boxPoints(rect2)
        box2 = np.intp(box2)
        # cv2.drawContours(image,[box2],0,(255,0,0),1)

        # Create a black mask with the same size as the image
        mask = np.zeros_like(image)

        # Fill the rotated bounding box on the mask with white
        cv2.drawContours(mask, [box2], -1, (255), thickness=cv2.FILLED)

        # Apply the mask to the original grayscale image
        masked_grayscale = cv2.bitwise_and(image, image, mask=mask)

        # Convert the masked grayscale image to a 3-channel color image
        result = cv2.cvtColor(masked_grayscale, cv2.COLOR_GRAY2BGR)

        mask = np.zeros(result.shape[:2], dtype=np.uint8)
        

        # Sort out the countors for angiosomes
        def offset_contour_fixed(contour, offset):
            # Ensure contour is in the correct shape (n_points, 2)
            contour = contour.reshape(-1, 2)
            contour = np.array(contour, dtype=np.float32)  

            # Compute normals for each point
            normals = []
            for i in range(len(contour)):
                # Get the current point, previous point, and next point (cyclically)
                p1 = contour[i]
                p0 = contour[(i - 1) % len(contour)]  # Previous point
                p2 = contour[(i + 1) % len(contour)]  # Next point

                # Edge vectors
                e1 = p1 - p0
                e2 = p2 - p1

                # Compute normals for both edges
                n1 = np.array([-e1[1], e1[0]]) / np.linalg.norm(e1)
                n2 = np.array([-e2[1], e2[0]]) / np.linalg.norm(e2)

                # Average normals at the current point
                normal = (n1 + n2) / np.linalg.norm(n1 + n2)
                normals.append(normal)

            # Offset the contour inward by fixed pixels
            normals = np.array(normals)
            offset_points = contour - normals * offset  # Inward offset

            return offset_points.astype(np.int32)


        # Offset the foot's contour inward by 4 pixels
        cnt3 = offset_contour_fixed(separate_left_right.cnt2, 4)  
        
        if 'separate_left_right.cnt1' in locals():
            cnt_left = offset_contour_fixed(separate_left_right.cnt1, 4)
        else:
            cnt_left = None
            

        # Draw the contour on the mask
        cv2.drawContours(mask, [cnt3], -1, 255, thickness=cv2.FILLED)     # uncomment to show miniature contour
        

        # Create an empty result image with the same shape as the original
        final_result = np.zeros_like(result)
        cv2.drawContours(final_result, [separate_left_right.cnt2], -1, (255, 255, 255), 1)
        

        polygons = [(vertices1, (0, 0, 255)),    # Red, option: brown (0, 102, 204) 
                    (vertices2, (255, 0, 0)),    # Blue,         light Blue (255, 153, 51) 
                    (vertices3, (0, 128, 0)),    # Green
                    (vertices4, (128, 0, 128)),  # Purple            
                    (vertices5, (0, 165, 255))]  # Orange


        # Iterate through each polygon (aka angiosome)
        for vertices, color in polygons:
            # Create a mask for the current polygon
            poly_mask = np.zeros(result.shape[:2], dtype=np.uint8)
            cv2.fillPoly(poly_mask, [vertices], 255)
            
            # Combine the polygon mask with the contour mask
            combined_mask = cv2.bitwise_and(mask, poly_mask)

            # Apply the combined mask to the final result image with the color
            color_mask = np.zeros_like(result)
            color_mask[combined_mask > 0] = color

            # Accumulate the color masks
            final_result = cv2.add(final_result, color_mask)
        result = final_result

        
        # Initialize an empty list to store the 5 vectors
        max_pixel_value = []


        ii = 0
        if img_no == 1:
            old_miniature_mask = cnt3
        similarity_score = cv2.matchShapes(old_miniature_mask, cnt3, cv2.CONTOURS_MATCH_I1, 0.0)
        print('Similarity_score', similarity_score)
        process_frame = False
        if similarity_score <= 0.1:
            process_frame = True
        else:
            temp_evolution =  np.zeros_like(cv2.cvtColor(image, cv2.COLOR_GRAY2BGR))          
            with open(csv_file, mode='a', newline='') as file2:
                writer = csv.writer(file2)
                # Append selected frame number to CSV file
                writer.writerow([img_no])
            return np.zeros_like(image)


        for vertices, color in polygons:
            # Create a mask for the polygon area
            mask = np.zeros_like(temp_image)
            ii = ii + 1
            cv2.fillPoly(mask, [vertices], 255)
            mask2 = np.zeros_like(temp_image)  

            # Draw the contour on the mask
            cv2.drawContours(mask2, [cnt3], -1, 255, thickness=cv2.FILLED)    

            # Work out the resultant mask
            combined_mask = cv2.bitwise_and(mask, mask2)  
            
            # Extract pixel values within the polygon area
            masked_image = cv2.bitwise_and(temp_image, temp_image, mask=combined_mask)
            
            # Get the indexes of non-zero pixels  Added
            indexes = np.argwhere(masked_image > 0)  # Returns an array of (row, column) pairs

            # Extract values from matr3 using these indexes  Added
            masked_image = np.array([temp_image2[row, col] for row, col in indexes])

            # Filter pixel values within the range 20 to 30 degrees
            filtered_pixels = masked_image[(masked_image >= 15) & (masked_image <= 45)] 

            # Ensure there are no NaN values and convert to integers
            filtered_pixels = filtered_pixels[~np.isnan(filtered_pixels)].astype(np.float32)  
            print('max temp type', type(filtered_pixels) )
            
            # Compare last two contours
            if len(filtered_pixels) > 0:
                if np.max(filtered_pixels) > 0 and pass2 == True and process_frame == True: 
                    
                    if  img_no != 1 and abs(np.max(filtered_pixels) - globals()[f'y{ii}'][-1]) >= 0.5:
                        max_pixel_value.append(globals()[f'y{ii}'][-1])
                        data[ii-1].append(globals()[f'y{ii}'][-1])
                    else:
                        max_pixel_value.append(np.max(filtered_pixels))
                        data[ii-1].append(np.max(filtered_pixels))
                else:
                    max_pixel_value.append(old_max_pixel_value[len(max_pixel_value)-1])
                    data[ii-1].append(old_max_pixel_value[len(max_pixel_value)-1])
                print(f"Maximum pixel value in the range for polygon {ii}: {np.max(filtered_pixels)}")
            else:
                print(f"No pixel values found in the range for polygon {ii}.")
                max_pixel_value.append(old_max_pixel_value[len(max_pixel_value)-1])
                data[ii-1].append(old_max_pixel_value[len(max_pixel_value)-1])

        # Memorize variables for next iteration reference
        old_miniature_mask = cnt3
        old_max_pixel_value = max_pixel_value


    # Fix first image issues
    if  old_cy_1 == 0 and old_cy_2 == 0 and (pass2 == True or pass2 == True):                           
        try:
            if cv2.contourArea(separate_left_right.cnt1) < 12000:
                old_cy_1 = separate_left_right.cy1
                go_1 = True
        except Exception as e:
            pass
        try:
            if cv2.contourArea(separate_left_right.cnt2) < 12000:
                old_cy_2 = separate_left_right.cy2
                go_2 = True
        except Exception as e:
            pass
    

    # Plot feet angiosomes' updated max Temp
    # X-axis
    try:
        x_axis
    except NameError:
        x_axis = []


    # Y-axis
    try:
        y
    except NameError:
        y = [[] for _ in range(5)]


    x_axis.append(img_no)
    for j in range(5):
        y[j].append(max_pixel_value[j])
    
    
    # Return frame with angiosomes
    if pass2 == True:                                             
        return result #result / image                                           
    else:
        result = np.zeros_like(result)
        with open(csv_file, mode='a', newline='') as file2:
            writer = csv.writer(file2)
            # Append selected frame number to CSV file
            writer.writerow([img_no]) 
        return result


# 'Main' to run as standalone
if __name__ == '__main__':

    # Test execution time
    start_time_frame = datetime.now() 

    contours = image = temp_image = img_no = csv_file = None
    result = get_bounding_box(contours, image, temp_image, img_no, csv_file, number_of_boxes=2)

    # Print processing time
    duration = datetime.now() - start_time_frame
    print('Processing_angiosomes_fig_time = ', 1000 * duration.total_seconds(), '[ms]')
    
