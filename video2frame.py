import os
import numpy as np

import cv2
from tqdm import tqdm

def create_folder(folder):
    if not os.path.exists(folder):
        os.makedirs(folder)

def video_to_frame(video_path):
    save_path = os.path.join('./')

    for i in range(2):
        #create two folders to save frames
        path = os.path.join(save_path, 'frames', 'group{}'.format(i))
        create_folder(path)

    video = cv2.imreadmulti(video_path, flags = cv2.IMREAD_UNCHANGED)
    for index, frame in tqdm(enumerate(video[1])):
        frame = np.uint8(frame)
        frame = frame[-400:,:]
        if index % 2 == 0:
            cv2.imwrite(os.path.join(save_path,'frames','group0','%.04d.tif' % (index//2)), frame)
        else:
            cv2.imwrite(os.path.join(save_path,'frames','group1','pair_%.04d.tif' % (index//2)), frame)

def normalization(array, maximum = 16376.0):
    return (array) * 255.0 / maximum

if __name__ == "__main__":
    video_path = "DL08_OD_1ART_UNFOCUS.tif"

    video_to_frame(video_path)