import cv2
import matplotlib.pyplot as plt
import copy
import numpy as np
from src import model
from src import util
from src.body import Body
from src.hand import Hand
import time
import math
import os
from glob import glob
from tqdm import tqdm
import pdb

body_estimation = Body('model/body_pose_model2.pth')

# Specify the folder path containing long mp4 files
mp4_source_folder_path = '../../Dataset_CVDLPT_Videos_Segments_11_2023'

# Specify the folder path to store the npz files
npz_destination_folder_path = './output'

# Loop through files in the folder
for filename in os.listdir(mp4_source_folder_path):
    if filename.endswith('.mp4'):  # Check if the file is an mp4 file
        npz_destination_file_path = os.path.join(npz_destination_folder_path,f"{filename.split('.')[0]}_2D.npz")
        mp4_source_file_path = os.path.join(mp4_source_folder_path,filename)
        if not os.path.exists(npz_destination_file_path):
            print(npz_destination_file_path)
            print(filename)
            
            pdb.set_trace()

            cap = cv2.VideoCapture(mp4_source_file_path)

            cnt = 0

            # Get the total number of frames
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            # Display the total number of frames
            print(f"Total number of frames in the video: {total_frames}")
           
            pbar = tqdm(total=total_frames)

            while(cap.isOpened()):
                 ret, frame = cap.read()

                 if frame is None:
                    break
                 
                 candidate, subset = body_estimation(frame)
                 pbar.update(1)

                 npzFrame = np.ones((1,18,2))*-1
                 for i in range(18):
                     for n in range(len(subset)):
                         index = int(subset[n][i])
                         if index < 0 or index > 17 :
                             continue
                         #npzFrame[0,index] = candidate[index][0:2]
                         npzFrame[0,i] = candidate[index][0:2]
                 if cnt == 0:
                     npz = npzFrame
                 else:
                     npz = np.concatenate((npz,npzFrame),axis=0)

                 cnt=cnt+1

            pbar.close()
            cap.release()
           
            # Save the array into an npz file
            np.savez(npz_destination_file_path, npz)

