#!/usr/bin/python

'''
Main script to execute the foot-tracking app by calling the modules

by Rafal June 2025
''' 

# Prepare compute environment
import os 
import csv
import skimage
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import cv2
#cv2.setNumThreads(0) 
import numpy as np
import torch
print(torch.version.cuda)
print(torch.cuda.get_device_name())

# Check if GPU is available
#device = "cpu"
device = "cuda" if torch.cuda.is_available() else "cpu"
if device == "cuda":
    num_workers = torch.cuda.device_count() * 8
    #print('num_workers', num_workers)
print(device)
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data import RandomSampler
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torchvision
import torchmetrics
import albumentations as A
from albumentations.pytorch import ToTensorV2
import pytorch_lightning as pl
from torchmetrics import JaccardIndex
from torchmetrics.functional import accuracy
from pytorch_lightning.profilers import AdvancedProfiler
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint, EarlyStopping
#from lightning.pytorch.callbacks import EarlyStopping
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from unet import UNet
from datetime import datetime
import math
import time
import tifffile
import argparse
from PIL import Image
import gc
from prediction_singled_batch import prediction
from get_contour import contour
import get_bounding_box 
import plot_temp_tracking_figure
import get_percentage_ratios
from get_isotherm_percentage_figure import get_percentage_figure
#png_image.astype(np.float32)  # Reduce memory size
torch.cuda.empty_cache()
torch.set_float32_matmul_precision('medium')


# Helper function for predictions visualisation
def visualize(**images):
    global save_folder, img_no
    for name, image in images.items():
        # Convert PyTorch tensor to NumPy array if needed
        if isinstance(image, torch.Tensor):
            image = image.cpu().numpy()  # Move to CPU and convert to NumPy array
        
        # Ensure image values are scaled to 0-255 for saving as uint8 format
        grayscale_image = (image * 255).astype(np.uint8)

        # Save the image using Pillow in single-channel grayscale format
        if name == 'image':
            save_path = os.path.join(save_folder, f"images/{name}_{img_no}.tif")    # or  .png
        elif name == 'prediction':
            save_path = os.path.join(save_folder, f"masks/{name}_{img_no}.tif")     # or  .png

        # Convert the NumPy array to a PIL Image and save in grayscale mode ('L')
        Image.fromarray(grayscale_image).save(save_path, format='TIFF')             # or "PNG"


# Main
if __name__ == '__main__':
    # Time runtime
    start_time = datetime.now() 


    # Create a folder (if doesn't already exist) to save the pair: original image-predicted mask
    save_folder = "./images_&_masks_for_GUI/"
    os.makedirs(save_folder, exist_ok=True)
    os.makedirs(f'{save_folder}/images/', exist_ok=True)
    os.makedirs(f'{save_folder}/masks/', exist_ok=True)
 

    #parser = argparse.ArgumentParser(description="Process two input parameters")
    #parser.add_argument("--param1", type=str, help="patient # value")
    #parser.add_argument("--param2", type=str, help="segment # value")
    #args = parser.parse_args()

    patient =  '016' #args.param1 
    flir_no = '0884' #args.param2 
    #path = f'./output_images/for_video_{patient}_{flir_no}_2freq/'
    #

    #try:
    #    os.makedirs(path, exist_ok=True)  # 'exist_ok=True' avoids errors if already exists 
    #    print(f"Directory created: {path}")
    #except Exception as e:
    #    print(f"An error occurred: {e}")
    
 
    # Initialize variables to store percentages for each color
    img_no = 1
    img_no_start = 1
    x_axis_percent = []
    old_cy_1 = 0
    old_cy_2 = 0
    old_area_cnt1 = 0
    old_area_cnt2 = 0
    max_pixel_value = []
    outlier_count = 0
   

    # File path for disjointed frames log
    csv_file =  f'FLIR{flir_no}_foulty_frames.csv'


    # Load temperature map images for dev
    #ids2 = os.listdir(f'./data/{patient}_{flir_no}_2freq_temp/images/')
    ids2 = os.listdir(f'../../data/{patient}_{flir_no}_2freq_temp/images/')
    ids2 = sorted(ids2, key=lambda x: int(os.path.splitext(x)[0]))
    temp_imgs2 = [cv2.imread(os.path.join(f'./data/{patient}_{flir_no}_2freq_temp/images/', i), cv2.IMREAD_UNCHANGED) for i in ids2]


    # Load visual images for dev
    ids  = os.listdir(f'../../data/{patient}_{flir_no}_2freq/images/')  
    ids = sorted(ids, key=lambda x: int(os.path.splitext(x)[0]))
    temp_imgs = temp_imgs_bright = [os.path.join(f'./data/{patient}_{flir_no}_2freq/images/', i) for i in ids] 


    # Preppare a file to store disjointed frame numbers
    with open(csv_file, mode='a', newline='') as file2:
        writer = csv.writer(file2)
        # Write header if the file is empty
        file2.seek(0, 2)  # Move to end of file
        if file2.tell() == 0:
            writer.writerow(["Bad Frame Number"])


    # Load images for PRODUCTION!
    #images_folder_path = f'./data/'           # <- data folder (placeholder) containing input images
    #ids  = os.listdir(images_folder_path)                                      
    #ids = sorted(ids, key=lambda x: int(os.path.splitext(x)[0]))
    #images = [os.path.join(images_folder_path, i) for i in ids] 


    for img in temp_imgs: #images:
        # Time frame processing time
        start_time_frame = datetime.now() 


        # Get predictions from current 'img' frame
        predicted_mask = prediction(img)


        # Infer contours
        contours, png_image = contour(predicted_mask)
        
        # Imread visual frame for logical processing
        temp_image = cv2.imread(temp_imgs[img_no-img_no_start], 0)   
        # Imread temp. map frame    
        temp_image2 = temp_imgs2[img_no-img_no_start]    
        # Imread and condition visual frame for adding colored isotherms
        temp_image_bright_gray = cv2.imread(temp_imgs_bright[img_no-img_no_start], 0)   
        temp_image_bright = cv2.cvtColor(temp_image_bright_gray, cv2.COLOR_GRAY2BGR)
        

        if len(contours) >= 1:  
            png_image = get_bounding_box.get_bounding_box(contours, png_image, temp_image, temp_image2, img_no, csv_file)                                                      
             
            # Reduce memory size
            png_image = png_image.astype(np.float32)                # <-remedy cache memory
            print("Number of dimensions1:", png_image.ndim)
            if png_image.ndim <= 2:            #???png_image.all() == None:                                          
                png_image  = np.zeros_like(cv2.cvtColor(png_image, cv2.COLOR_GRAY2BGR)) 
                outlier_count += 1
                #with open(csv_file, mode='a', newline='') as file2:
                #    writer = csv.writer(file2)
                #    # Append selected frame number to CSV file
                #    writer.writerow([img_no])  
                print("Number of dimensions2:", png_image.ndim)
            else:
                old_png_image = png_image 

        elif len(contours) == 0:
            png_image  = np.zeros_like(cv2.cvtColor(png_image, cv2.COLOR_GRAY2BGR)) 
            outlier_count += 1
            with open(csv_file, mode='a', newline='') as file2:
                writer = csv.writer(file2)
                # Append selected frame number to CSV file
                writer.writerow([img_no]) 


            # Start and end points for the arrow
            arrow_length = 50 
            start_point = (get_bounding_box.cx_arrow, 240 - 5)  # Bottom edge of the image
            end_point = (get_bounding_box.cx_arrow, 240 - 1 - arrow_length)  # Pointing upward

        # Draw the arrow indicating the foot under processing
        cv2.arrowedLine(temp_image_bright, start_point, end_point, (255, 255, 255), 2, tipLength=0.2)


        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(temp_image_bright,f'{img_no}',(150,30), font, 1, (255,255,255), 1, cv2.LINE_AA)   
        cv2.putText(png_image,f'Right Foot',(210,40), font, 0.5, (255,255,255), 1, cv2.LINE_AA)  
        cv2.putText(png_image,f'RED-LP',(210,80), font, 0.3, (255,255,255), 1, cv2.LINE_AA)  
        cv2.putText(png_image,f'BLUE-MP',(210,110), font, 0.3, (255,255,255), 1, cv2.LINE_AA)  
        cv2.putText(png_image,f'GREEN-LC',(210,140), font, 0.3, (255,255,255), 1, cv2.LINE_AA)  
        cv2.putText(png_image,f'PURPLE-MC',(210,170), font, 0.3, (255,255,255), 1, cv2.LINE_AA)  
        cv2.putText(png_image,f'ORANGE-GT',(210,200), font, 0.3, (255,255,255), 1, cv2.LINE_AA)  
        

        # Prepare temperature evolution figure
        plot_temp_tracking_figure.plot_series()


        if np.all(png_image == 0):
            # Read-in the saved temperature evolution figure using OpenCV
            temp_evolution =  np.zeros_like(cv2.cvtColor(png_image, cv2.COLOR_GRAY2BGR))
        else:
            temp_evolution = cv2.imread('temporary_temp_evolution.png') 

         
        png_image_norm = skimage.exposure.rescale_intensity(png_image, in_range='image', out_range=(0, 255)).astype(np.uint8)#@@@
        temp_image_norm = skimage.exposure.rescale_intensity(temp_image_bright, in_range='image', out_range=(0, 255)).astype(np.uint8)
        temp_evolution = cv2.resize(temp_evolution, (320, 240)) 
        temp_evolution_norm = skimage.exposure.rescale_intensity(temp_evolution, in_range='image', out_range=(0, 255)).astype(np.uint8)
        

        # Percentage ratio maps processing
        if get_bounding_box.process_frame == True:
            if get_bounding_box.cnt_left is not None:
               temp_image_bright = get_percentage_ratios.get_percentage_maps(temp_image2, temp_image_bright_gray, temp_image_bright, img_no, get_bounding_box.cnt3, get_bounding_box.cnt_left)
            else:
                temp_image_bright = get_percentage_ratios.get_percentage_maps(temp_image2, temp_image_bright_gray, temp_image_bright, img_no, get_bounding_box.cnt3)
            
            # Pecentage maps figure comes here
            get_percentage_figure(get_bounding_box.x_axis, get_percentage_ratios.white_ratios, get_percentage_ratios.yellow_ratios, get_percentage_ratios.orange_ratios, get_percentage_ratios.red_ratios, get_percentage_ratios.blue_ratios, get_percentage_ratios.rounded_max_value)

            # Read the saved figure using OpenCV and optimize resolution
            temp_file2 = 'temporary_percent_evolution.png'
            percent_evolution = cv2.imread(temp_file2)
        else:
                dummy_image = np.zeros((240, 320), dtype=np.uint8)
                percent_evolution =  np.zeros_like(cv2.cvtColor(dummy_image, cv2.COLOR_GRAY2BGR))
        percent_evolution = cv2.resize(percent_evolution, (320, 240))
        percent_evolution_norm = skimage.exposure.rescale_intensity(percent_evolution, in_range='image', out_range=(0, 255)).astype(np.uint8)
      

        # Generate isotherm areas color mapped feet image for dashboard 
        plt.figure(figsize=(8, 6))
        plt.imshow(temp_image_bright)
        plt.title('Frames (every second)', fontsize=22)
        # Remove all axes
        plt.axis('off')
        #plt.show(block=False)  # Ensure the figures don't block execution
        # Save the figure to a temporary file
        temp_file3 = 'temporary_color_mapped_frame.png'
        plt.savefig(temp_file3)
        # Read the saved figure using OpenCV
        color_mapped_image = cv2.imread(temp_file3)


        # Generate grand truth image for dashboard reference
        plt.figure(figsize=(8, 6))
        plt.imshow(temp_image_bright_gray)
        plt.title('Ground truth', fontsize=22)
        plt.axis('off')
        temp_file4 = 'temporary_original_frame.png'
        plt.savefig(temp_file4)
        original_image = cv2.imread(temp_file4, 0)
        original_image = cv2.cvtColor(original_image, cv2.COLOR_GRAY2BGR)
        

        # Function to add white padding
        def add_white_padding(image, target_height, target_width):
            original_height, original_width = image.shape[:2]
            top_padding = (target_height - original_height) // 2  # Padding above
            bottom_padding = target_height - original_height - top_padding  # Padding below
            left_padding = (target_width - original_width) // 2  # Padding left (if width differs)
            right_padding = target_width - original_width - left_padding  # Padding right
            # Add white (255) padding around the image
            padded_image = cv2.copyMakeBorder(
                image,
                top_padding,
                bottom_padding,
                left_padding,
                right_padding,
                cv2.BORDER_CONSTANT,
                # White color
                value=[255, 255, 255])
            return padded_image
        
        
        # Resize left column images (no padding needed)
        color_mapped_image = cv2.resize(color_mapped_image, (320, 240))
        original_image = cv2.resize(original_image, (320, 240))
        png_image_norm = cv2.resize(png_image_norm, (320, 240))
        left_column = np.vstack([color_mapped_image, original_image, png_image_norm])  # Stack as is
        
        # Resize right column images (add padding to preserve aspect ratio)
        percent_evolution_norm = cv2.resize(percent_evolution_norm, (320, 240))  # Resize first
        temp_evolution_norm = cv2.resize(temp_evolution_norm, (320, 240))        # Resize first
        percent_evolution_norm = add_white_padding(percent_evolution_norm, 360, 320)
        temp_evolution_norm = add_white_padding(temp_evolution_norm, 360, 320)
        
        # Create the right column by stacking padded images vertically
        right_column = np.vstack([percent_evolution_norm, temp_evolution_norm])
        
        # Combine the columns horizontally
        dashboard = np.hstack([left_column, right_column])
        
        # Save the final dashboard to file
        #filename = f'./medical_imaging/images/for_video_test15/{img_no}.png'
        filename = f'../medical_imaging/images/for_video_test15/{img_no}.png'
        cv2.imwrite(filename, dashboard)
                             
                                                 
        # Clear Matplotlib figures to free memory
        plt.close('all')                     
       

        if temp_imgs[img_no-img_no_start] == temp_imgs[-1]:
            print('frames_no, outlier_no, ratio', img_no - img_no_start + 1, outlier_count, outlier_count/(img_no - img_no_start + 1) )
       

        img_no += 1
        time_frame_duration = datetime.now() - start_time
        print('frame_time =', time_frame_duration.total_seconds(), 'fps =', 1/time_frame_duration.total_seconds())


        # Delete large objects
        del png_image  
        torch.cuda.empty_cache()  
        # Force garbage collection
        gc.collect()   
       

    # Dictionary to store standard deviations    
    std_devs = {}  

    # Loop through y1 to y5 angiosome areas
    for i in range(5):  
        std_dev = np.std(get_bounding_box.y[i])  # Compute standard deviation
        std_devs[f'y{i+1}'] = std_dev  # Store it in the dictionary


    # Print the standard deviations
    for variable, std_dev in std_devs.items():
        print(f'Standard deviation of {variable}: {std_dev}')


    # Save the 5 vectors to a CSV file
    with open(f'FLIR{flir_no}.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        # Optional headers
        writer.writerow(['LP', 'MP', 'LC', 'MC', 'GT'])  
        # Transpose the data so that each row corresponds to a single iteration
        writer.writerows(zip(*get_bounding_box.data)) 


    # Define your variable names and values to save
    time_duration = datetime.now() - start_time_frame
    variable_names = ['seg_no', 'seconds_to_process', 'fps', 'frames_no', 'outlier_no', 'ratio', 'LP_SD', 'MP_SD', 'LC_SD', 'MC_SD', 'GT_SD']
    values = [
        flir_no,
        time_duration.total_seconds(),
        len(temp_imgs2) / time_duration.total_seconds(),
        img_no - img_no_start,
        outlier_count,
        outlier_count / (img_no - img_no_start),
        std_devs['y1'], std_devs['y2'], std_devs['y3'], std_devs['y4'], std_devs['y5']]


    # Specify the output CSV file name
    output_file = 'output.csv'


    # Check if the file exists
    file_exists = os.path.isfile(output_file)


    # Open the file in append mode ('a')
    with open(output_file, mode='a', newline='') as file:
        writer = csv.writer(file)

        # If the file doesn't exist or is empty, write the header first
        if not file_exists or os.path.getsize(output_file) == 0:
            writer.writerow(variable_names)

        # Append the values as a new row
        writer.writerow(values)


    print(f"Stats data has been saved to {output_file}")
    print("Angiosomes data saved to segm. named .csv")       
    cv2.destroyAllWindows()