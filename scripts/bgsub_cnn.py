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

sys.path.append("/home/leo/Desktop/vgg_segmentation")
#import loaders
#import utils
import segnet
#import losses_scores
#import trainer
#import my_transforms
#from torch import optim
from torch.autograd import Variable
import torch

__author__ = "Leonardo Citraro"
__email__ = "leonardo.citraro@epfl.ch"
    
def main(input_folder = ".",
         output_folder = ".",
         sigma = 1.0,
         cnn_params = "network.pickle",
         threshold = 0.2):
    """ Compute simple background substraction.
    
    Parameters
    ----------

    """
    
    print("input_folder:", input_folder)
    print("output_folder:", output_folder)
    
    filenames = utils.find_images(input_folder, "*") 
    print("Found {} images".format(len(filenames)))
    
    utils.mkdir(output_folder)
    
    model = segnet.VGGSeg(trunc_block='conv3_56', mlp_depth=2, n_classes=2, upsampling="nearest")
    model.load_state(cnn_params)
    
    for i, filename in enumerate(filenames):
    
        print("Processing image {}..".format(filename))
    
        img = np.float32(imageio.imread(filename))/255.0
        pred = model(Variable(torch.from_numpy(img.transpose(2,0,1)[np.newaxis,:]))).data.numpy()[0,1]
        
        pred = np.uint8((pred>threshold)*255)
        
        utils.save_image(os.path.join(output_folder, "bg_{}.png".format(i)), pred)    
    
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--input_folder", "-i", type=str, required=True)
    parser.add_argument("--output_folder", "-o", type=str, required=True)
    parser.add_argument("--sigma", "-s", type=float, default=0.0, required=False)
    parser.add_argument("--cnn_params", "-p", type=str, default="network.pickle", required=True)
    parser.add_argument("--threshold", "-t", type=float, default=0.2, required=True)

    args = parser.parse_args()
    main(**vars(args)) 
        
