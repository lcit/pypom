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

class BackgroundSubstractionAVG:
    def __init__(self, alpha, threshold):
        self.alpha  = alpha
        self.threshold = threshold
        self.backGroundModel = None

    def apply(self, frame):
        if self.backGroundModel is None:
            self.backGroundModel =  frame
        else:
            self.backGroundModel =  frame * self.alpha + self.backGroundModel * (1 - self.alpha)

        return np.uint8(np.logical_or.reduce(cv2.absdiff(self.backGroundModel.astype(np.uint8), frame) > self.threshold, 2)*255)
    
def main(input_folder = ".",
         output_folder = ".",
         sigma = 1.0,
         alpha = 0.5,
         threshold=32):
    """ Compute simple background substraction.
    
    Parameters
    ----------
    input_folder: str
        path to folder with the images
    output_folder : str
        path where to save the resulting images 
    sigma : float
        Gaussian blurring sigma applied to the image before entering the subtractor
    alpha : float [0,1]
        Parameter affecting the learning rate of the backgorund
    threshold : int
        Threshold used to separate background from foregraound after abs-substraction
    """
    
    print("input_folder:", input_folder)
    print("output_folder:", output_folder)
    
    filenames = utils.find_images(input_folder, "*") 
    print("Found {} images".format(len(filenames)))
    
    utils.mkdir(output_folder)
    
    model = BackgroundSubstractionAVG(alpha, threshold)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    
    for i, filename in enumerate(filenames):
    
        print("Processing image {}..".format(filename))
    
        img = utils.load_image(filename)

        img = np.uint8(utils.gaussian_blurring(img, sigma))
        
        pred = model.apply(img)
        pred = cv2.morphologyEx(pred, cv2.MORPH_OPEN, kernel)  
        
        utils.save_image(os.path.join(output_folder, "bg_{}.png".format(i)), pred)    
    
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--input_folder", "-i", type=str, required=True)
    parser.add_argument("--output_folder", "-o", type=str, required=True)
    parser.add_argument("--sigma", "-s", type=float, default=1.0, required=False)
    parser.add_argument("--alpha", "-a", type=float, default=0.5, required=False)
    parser.add_argument("--threshold", "-t", type=int, default=32, required=False)

    args = parser.parse_args()
    main(**vars(args)) 
        
