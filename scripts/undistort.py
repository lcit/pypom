import cv2
import numpy as np
import os
import argparse
import sys
import inspect

from pypom import utils

__author__ = "Leonardo Citraro"

def main(input_folder="",
        output_folder="",
        intrinsics_json="intrinsics.json",
        downsampling=1):

    print("input_folder:", input_folder)
    print("output_folder:", output_folder)
    
    filenames = utils.find_images(input_folder, "*") 
    print("Found {} images".format(len(filenames)))
    
    utils.mkdir(output_folder)
    
    K, dist, _ = utils.retrieve_intrinsics_from_json(intrinsics_json)
    K = utils.scale_homography(K, 1.0/downsampling)


    for filename in filenames:

        img = utils.load_image(filename)
        h, w = img.shape[:2]
        
        newcameramtx, roi = cv2.getOptimalNewCameraMatrix(K, dist, (w, h), 0.5, (w, h), centerPrincipalPoint=True)
        
        print("Processing image {} -  roi={}".format(os.path.basename(filename), roi))

        dst = cv2.undistort(img, K, dist, None, newcameramtx)

        # crop the image
        x,y,w,h = roi
        dst = dst[y:y+h, x:x+w]

        utils.save_image(os.path.join(output_folder, os.path.basename(filename)), dst)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--input_folder", "-i", type=str, required=True)
    parser.add_argument("--output_folder", "-o", type=str, required=True)
    parser.add_argument("--intrinsics_json", "-j", type=str, required=True)
    parser.add_argument("--downsampling", "-d", type=int, default=1, required=False)

    args = parser.parse_args()
    main(**vars(args)) 
        
    
