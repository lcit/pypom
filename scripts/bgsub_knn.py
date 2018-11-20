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
import sys
import inspect

this_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(this_dir)
sys.path.append(parent_dir)
from pypom import utils

__author__ = "Leonardo Citraro"
__email__ = "leonardo.citraro@epfl.ch"
    
def main(input_folder = ".",
         output_folder = ".",
         sigma = 1.0,
         history = 500,
         dist2Threshold = 400.0):
    """ Compute simple background substraction.
    
    Parameters
    ----------
    input_folder: str
        path to folder with the images
    output_folder : str
        path where to save the resulting images 
    sigma : float
        Gaussian blurring sigma applied to the image before entering the subtractor
    history : int
        Opencv KNN backgrond sub history
    dist2Threshold : int
        Opencv KNN backgrond sub dist2Threshold
    """
    
    print("input_folder:", input_folder)
    print("output_folder:", output_folder)
    
    filenames = utils.find_images(input_folder, "*") 
    print("Found {} images".format(len(filenames)))
    
    utils.mkdir(output_folder)
    
    model = cv2.createBackgroundSubtractorKNN(history, dist2Threshold, detectShadows = True)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    
    for i, filename in enumerate(filenames):
    
        print("{} -- Processing image {}..".format(i, filename))
    
        img = utils.load_image(filename)

        img = np.uint8(utils.gaussian_blurring(img, sigma))
        
        pred = model.apply(img)
        pred = cv2.morphologyEx(pred, cv2.MORPH_OPEN, kernel)  
        pred = np.uint8((pred==255)*255) # this removes the shadow
        
        utils.save_image(os.path.join(output_folder, "bg_{}.jpg".format(i)), pred)    
    
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--input_folder", "-i", type=str, required=True)
    parser.add_argument("--output_folder", "-o", type=str, required=True)
    parser.add_argument("--sigma", "-s", type=float, default=1.0, required=False)
    parser.add_argument("--history", "-y", type=int, default=500, required=False)
    parser.add_argument("--dist2Threshold", "-t", type=int, default=400, required=False)

    args = parser.parse_args()
    main(**vars(args)) 
        
