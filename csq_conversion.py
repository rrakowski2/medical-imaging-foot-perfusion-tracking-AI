#!/usr/bin/python

'''
Script to convert thermal camera native .csq file (aided by a Perl executable adopted from: https://github.com/AlexanderProd/csq?tab=readme-ov-file) to collections of tif and png image frames for the foot-tracking app
by Rafal Apr 2025
''' 


# Set compute environment
import cv2
import gc
import os
import numpy as np
import re
import tempfile
import subprocess
import exiftool
#print("PyExifTool is successfully installed!")
from libjpeg import decode
#print("libjpeg is working!")
from numpy import exp, sqrt, log
import csq
from csq import CSQReader
from datetime import datetime
import pickle  # To store and retrieve the last processed index


# Define the folder for checkpoints
checkpoint_folder = 'checkpoints/'
os.makedirs(checkpoint_folder, exist_ok=True)  # Ensure checkpoint folder exists


# Function to load the last processed file name
def load_file_checkpoint():
    file_checkpoint_path = os.path.join(checkpoint_folder, 'file_checkpoint.pkl')
    if os.path.exists(file_checkpoint_path):
        with open(file_checkpoint_path, 'rb') as f:
            return pickle.load(f)
    return None  # Start from the beginning if no file checkpoint exists


# Function to save the last processed file name
def save_file_checkpoint(file_name):
    file_checkpoint_path = os.path.join(checkpoint_folder, 'file_checkpoint.pkl')
    with open(file_checkpoint_path, 'wb') as f:
        pickle.dump(file_name, f)


# Function to load the last frame checkpoint for a specific file
def load_frame_checkpoint(file_name):
    frame_checkpoint_path = os.path.join(checkpoint_folder, f'{file_name}.pkl')
    if os.path.exists(frame_checkpoint_path):
        with open(frame_checkpoint_path, 'rb') as f:
            return pickle.load(f)
    return 0  # Start from the beginning if no frame checkpoint exists


# Function to save the frame checkpoint for a specific file
def save_frame_checkpoint(file_name, index):
    frame_checkpoint_path = os.path.join(checkpoint_folder, f'{file_name}.pkl')
    with open(frame_checkpoint_path, 'wb') as f:
        pickle.dump(index, f)


# Function to erase checkpoints for a specific file
def erase_file_and_frame_checkpoint(file_name):
    file_checkpoint_path = os.path.join(checkpoint_folder, 'file_checkpoint.pkl')
    frame_checkpoint_path = os.path.join(checkpoint_folder, f'{file_name}.pkl')
    
    if os.path.exists(file_checkpoint_path):
        os.remove(file_checkpoint_path)
    if os.path.exists(frame_checkpoint_path):
        os.remove(frame_checkpoint_path)


# Main variables
main_folder = '22_patients'
start_time = datetime.now()


# Load the checkpoint for file iteration
last_file_checkpoint = load_file_checkpoint()
skip_to_file = last_file_checkpoint is not None  # Flag to skip already processed files


# Loop through all subfolders and files
for subfolder in os.listdir(main_folder):
    subfolder_path = os.path.join(main_folder, subfolder)
    
    
    # Check if it's a directory
    if os.path.isdir(subfolder_path):
        for file_name in os.listdir(subfolder_path):
            if file_name.endswith('.csq'):  # Check for .csq video files
                # Skip already processed files if resuming
                if skip_to_file:
                    if file_name == last_file_checkpoint:
                        skip_to_file = False  # Resume processing from this file
                    continue

                try:
                    # Save the file checkpoint at the start of processing
                    save_file_checkpoint(file_name)
                    
                    # Split the file name to extract FLIR number
                    parts = file_name.split('R')
                    flir_no = parts[1].split('.')[0]

                    # Specify and create nested directories (only if they don't exist)
                    nested_path1 = f'./data/{subfolder}_{flir_no}_2freq/'
                    nested_path2 = f'./data/{subfolder}_{flir_no}_2freq/images/'
                    nested_path3 = f'./data/{subfolder}_{flir_no}_2freq/masks/'
                    nested_path4 = f'./data/{subfolder}_{flir_no}_2freq_temp/images/'
                    
                    # Create folders only if not already present
                    if not os.path.exists(nested_path1): 
                        os.makedirs(nested_path1, exist_ok=True)
                        os.makedirs(nested_path2, exist_ok=True)
                        os.makedirs(nested_path3, exist_ok=True)
                        os.makedirs(nested_path4, exist_ok=True)
                        print(f"Directories created: {nested_path4}")

                    # Process the current file_name
                    video_path = os.path.join(subfolder_path, file_name)
                    print(f"Processing video: {video_path}")

                    reader = CSQReader(video_path)
                    total_frames = reader.count_frames()
                    print(f"Total frames in video: {total_frames}")

                    # Resume from the last processed frame or start from 0
                    i = load_frame_checkpoint(file_name)
                    print(f"Resuming from frame index: {i}")

                    while i < total_frames:
                        frame = reader.next_frame()
                        # Exit loop if there are no more frames
                        if frame is None: 
                            break

                        if i % 2 == 0:
                            filename1 = f'./data/{subfolder}_{flir_no}_2freq/images/{int(i/2)}.png'
                            filename2 = f'./data/{subfolder}_{flir_no}_2freq/masks/{int(i/2)}.png'

                            # Normalize to 8-bit
                            frame_8bit = cv2.normalize(frame, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

                            # Save the result
                            cv2.imwrite(filename1, frame_8bit)
                            cv2.imwrite(filename2, frame_8bit)

                            filename3 = f'./data/{subfolder}_{flir_no}_2freq_temp/images/{int(i/2)}.tif'

                            # Convert to 32-bit float
                            frame_32bit = frame.astype(np.float32)
                            cv2.imwrite(filename3, frame_32bit)

                            if i % 1000 == 0 and i != 0:
                                time_frame_duration = datetime.now() - start_time
                                print('iteration', i, '| frame_time =', time_frame_duration.total_seconds())

                            if i % 50 == 0:
                                save_frame_checkpoint(file_name, i)

                        i += 1

                    erase_file_and_frame_checkpoint(file_name)
                    print(f"Finished processing {video_path}. Checkpoints erased.")

                except Exception as file_error:
                    print(f"Error processing file {file_name}: {file_error}")
                    # Skip to the next file_name
                    continue  