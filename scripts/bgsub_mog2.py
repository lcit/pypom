#!/usr/bin/env python
""" Simple Opencv background substraction.
"""

import numpy as np
import cv2
import os
import imageio
import glob
import re
import argparse

from utils import find_images, mkdir

__author__ = "Leonardo Citraro"
__email__ = "leonardo.citraro@epfl.ch"
    
def main(input_folder = ".",
         output_folder = ".",
         history = 500,
         varThreshold = 16):
    """ Compute simple background substraction.
    
    Parameters
    ----------
    input_folder: str
        path to folder with the images
    output_folder : str
        path where to save the resulting images  
    history : int
        Opencv MOG2 backgrond sub history
    varThreshold : int
        Opencv MOG2 backgrond sub varThreshold
    """
    
    print("input_folder:", input_folder)
    print("output_folder:", output_folder)
    
    filenames = find_images(input_folder, "*") 
    print("Found {} images".format(len(filenames)))
    
    mkdir(output_folder)
    
    model = cv2.createBackgroundSubtractorMOG2(history, varThreshold, detectShadows = True)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    
    for i, filename in enumerate(filenames):
    
        print("Processing image {}..".format(filename))
    
        img = imageio.imread(filename)
        
        pred = model.apply(img)
        pred = cv2.morphologyEx(pred, cv2.MORPH_OPEN, kernel)    
        pred = np.uint8((pred==255)*255)
        
        imageio.imsave(os.path.join(output_folder, "bg_{}.png".format(i)), pred)    
    
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--input_folder", "-i", type=str, required=True)
    parser.add_argument("--output_folder", "-o", type=str, required=True)
    parser.add_argument("--history", "-y", type=int, default=500, required=False)
    parser.add_argument("--varThreshold", "-t", type=int, default=16, required=False)

    args = parser.parse_args()
    main(**vars(args)) 
        
