#!/usr/bin/env python3
import cv2
import argparse
import os
import sys
import inspect
import numpy as np

this_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(this_dir)
sys.path.append(parent_dir)
from pypom import utils

def main(input_folder="",
         output_folder="",
         a=1.0,
         b=0.0):
    """ 

    """
    
    print("input_folder:", input_folder)
    print("output_folder:", output_folder)
    print("a:", a)
    print("b:", b)

    filenames = utils.find_images(input_folder, "*") 
    print("Found {} images".format(len(filenames)))
    
    utils.mkdir(output_folder)

    for filename in filenames:
        
        print("Transformating image {}..".format(os.path.basename(filename)))

        img = utils.load_image(filename)

        img = np.float32(img)
        img = img*a+b
        img = np.uint8(np.around(img))
                    
        utils.save_image(os.path.join(output_folder, os.path.basename(filename)), img)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--input_folder", "-i", type=str, required=True)
    parser.add_argument("--output_folder", "-o", type=str, required=True)
    parser.add_argument("--a", "-a", type=float, default=1.0, required=True)
    parser.add_argument("--b", "-b", type=float, default=0.0, required=True)

    args = parser.parse_args()
    main(**vars(args))    
