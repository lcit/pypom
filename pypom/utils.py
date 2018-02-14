#!/usr/bin/env python
""" Utility function.
"""

import numpy as np
import cv2
import os
import imageio
import glob
import re
import argparse

__author__ = "Leonardo Citraro"
__email__ = "leonardo.citraro@epfl.ch"

def sort_nicely(l):
    """ Sort the given list in the way that humans expect.
    """
    convert = lambda text: int(text) if text.isdigit() else text
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ]
    return sorted(l, key=alphanum_key)

def mkdir(directory):
    directory = os.path.abspath(directory)
    if not os.path.exists(directory):
        os.makedirs(directory)
        
def find_images(file_or_folder, hint=None):  
    if hint is not None:
        file_or_folder = os.path.join(file_or_folder, hint)
    filenames = [f for f in glob.glob(file_or_folder)]
    filenames = sort_nicely(filenames)    
    filename_images = []
    for filename in filenames:
        if os.path.isfile(filename):
            _, extension = os.path.splitext(filename)
            if extension.lower() in [".jpg",".jpeg",".bmp",".tiff",".png",".gif"]:
                filename_images.append(filename)                 
    return filename_images 

def undistort(img, K, distCoeffs):   
    h, w = img.shape[:2]    
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(K, distCoeffs, (w, h), 0.0, (w, h), centerPrincipalPoint=True)    
    dst = cv2.undistort(img, K, distCoeffs, None, newcameramtx)

    x,y,w,h = roi
    dst = dst[y:y+h, x:x+w]

    return dst, newcameramtx, roi
