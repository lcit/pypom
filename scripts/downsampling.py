#!/usr/bin/env python3
import cv2
import argparse
import os
import sys
import inspect

from pypom import utils

__author__ = "Leonardo Citraro"

def main(input_folder="",
         output_folder="",
         downsampling=1,
         interpolation="nearest"):
    """ 

    """
    
    print("input_folder:", input_folder)
    print("output_folder:", output_folder)
    print("downsampling:", downsampling)
    print("interpolation:", interpolation)

    if interpolation == "nearest":
        interpolation = cv2.INTER_NEAREST
    elif interpolation == "linear":
        interpolation = cv2.INTER_LINEAR  
    elif interpolation == "cubic":
        interpolation = cv2.INTER_CUBIC 
    elif interpolation == "area":
        interpolation = cv2.INTER_AREA
    else:
        raise ValueError("Interpolation not understood.")
        sys.exit(0)
    
    filenames = utils.find_images(input_folder, "*") 
    print("Found {} images".format(len(filenames)))
    
    utils.mkdir(output_folder)

    for filename in filenames:
        
        print("Downsampling image {}..".format(os.path.basename(filename)))

        img = utils.load_image(filename)
        
        if downsampling > 1:
            img = cv2.resize(img, None, None, 1.0/downsampling, 1.0/downsampling, interpolation)
                    
        utils.save_image(os.path.join(output_folder, os.path.basename(filename)), img)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--input_folder", "-i", type=str, required=True)
    parser.add_argument("--output_folder", "-o", type=str, required=True)
    parser.add_argument("--downsampling", "-d", type=int, default=1, required=False)
    parser.add_argument("--interpolation", "-x", type=str, default="nearest", required=False)

    args = parser.parse_args()
    main(**vars(args))    
