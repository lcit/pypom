import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
import datetime
import sys
import inspect
import argparse
import multiprocessing
import json
import pickle
import glob
import re
import imageio
import shutil
from itertools import repeat

from pypom import utils

(cv2_major, cv2_minor, _) = cv2.__version__.split(".")
if int(cv2_major)<4:
    raise ImportError("Opencv version 4+ required!")

'''
ffmpeg -i VIDEO -r 0.5 frames/frame_%04d.jpg
python compute_intrinsics.py --folder_images ./frames -ich 6 -icw 8 -s 30 -t 24 --debug
https://markhedleyjones.com/storage/checkerboards/Checkerboard-A4-30mm-8x6.pdf
'''

__author__ = "Leonardo Citraro"

def process_image(filename_image, inner_corners_height, inner_corners_width, debug, debug_folder):
    print("Processing image {} ...".format(filename_image))

    gray = utils.rgb2gray(imageio.imread(filename_image))

    # Find the chess board corners
    ret, corners = cv2.findChessboardCorners(gray, (inner_corners_height,inner_corners_width),
                                             cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE)

    # If found, add object points, image points (after refining them)
    if ret == True:
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        imgp = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)

        if debug:
            gray = cv2.drawChessboardCorners(gray, (inner_corners_height,inner_corners_width), imgp, ret)
            imageio.imsave(os.path.join(debug_folder, os.path.basename(filename_image)), gray)

        return np.float32(imgp)
    return None

def main(folder_images, output_folder, description, 
         inner_corners_height, inner_corners_width, square_sizes, 
         alpha, threads, monotonic_range, debug):

    
    debug_folder = os.path.join(output_folder, "debug")
    undistorted_folder = os.path.join(output_folder, "undistorted")

    # delete if exist
    utils.rmdir(debug_folder)
    utils.rmdir(undistorted_folder)

    utils.mkdir(undistorted_folder)
    if debug:
        utils.mkdir(debug_folder)

    print("folder_images:", folder_images)
    print("description:", description)
    print("inner_corners_height:", inner_corners_height)
    print("inner_corners_width:", inner_corners_width)
    print("square_sizes:", square_sizes)
    print("alpha:", alpha)
    print("monotonic_range:", monotonic_range)
    print("threads:", threads)
    print("debug:", debug)

    current_datetime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    # prepare object points, like (0,0,0), (30,0,0), (60,0,0) ....
    # each square is 30x30mm
    # NB. the intrinsic parameters, rvec and distCoeffs do not depend upon the chessboard size, tvec does instead!
    objp = np.zeros((inner_corners_height*inner_corners_width,3), np.float32)
    objp[:,:2] = np.mgrid[0:inner_corners_height,0:inner_corners_width].T.reshape(-1,2)
    objp[:,:2] *= square_sizes

    filename_images = utils.find_images(folder_images, "*")
    if len(filename_images) == 0:
        print("!!! Unable to detect images in this folder !!!")
        sys.exit(0)
    print(filename_images)

    pool = multiprocessing.Pool(threads)
    res = pool.starmap(process_image, zip(filename_images, 
                                          repeat(inner_corners_height), 
                                          repeat(inner_corners_width), 
                                          repeat(debug), 
                                          repeat(debug_folder)))

    objpoints = [objp.copy() for r in res if r is not None] # 3d point in real world space
    imgpoints = [r.copy() for r in res if r is not None] # 2d points in image plane.

    img_shape = imageio.imread(filename_images[0]).shape[:2]
    
    print("working hard...")

    #ret, mtx, distCoeffs, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_shape[::-1], None, None)
    
    iFixedPoint = inner_corners_height-1
    ret, mtx, distCoeffs, rvecs, tvecs, newObjPoints, \
    stdDeviationsIntrinsics, stdDeviationsExtrinsics, \
    stdDeviationsObjPoints, perViewErrors = cv2.calibrateCameraROExtended(objpoints, imgpoints, img_shape[::-1],
                                                                          iFixedPoint, None, None)
    
    def reprojection_error(mtx, distCoeffs, rvecs, tvecs):
        # print reprojection error
        reproj_error = 0
        for i in range(len(objpoints)):
            imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, distCoeffs)
            reproj_error += cv2.norm(imgpoints[i],imgpoints2,cv2.NORM_L2)/len(imgpoints2)
        reproj_error /= len(objpoints) 
        return reproj_error
    
    reproj_error = reprojection_error(mtx, distCoeffs, rvecs, tvecs)
    print("RMS Reprojection Error: {}, Total Reprojection Error: {}".format(ret, reproj_error))
    
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, distCoeffs, img_shape[::-1], alpha, 
                                                      img_shape[::-1], centerPrincipalPoint=False)

    d_json = dict({"date":current_datetime, "description":description,
                   "K":mtx.tolist(), "K_new":newcameramtx.tolist(), "dist":distCoeffs.tolist(),
                   "reproj_error":reproj_error, "image_shape":img_shape})

    utils.json_write(os.path.join(output_folder, "intrinsics.json"), d_json)

    # The code from this pont on as the purpose of verifiying that the estimation went well.
    # images are undistorted using the compouted intrinsics
    
    # undistorting the images
    print("Saving undistorted images..")
    for i,filenames_image in enumerate(filename_images):

        img = imageio.imread(filenames_image)
        h, w = img.shape[:2]

        try:
            dst = cv2.undistort(img, mtx, distCoeffs, None, newcameramtx)
            # to project points on this undistorted image you need the following:
            # cv2.projectPoints(objpoints, rvec, tvec, newcameramtx, None)[0].reshape(-1,2)
            # or:
            # cv2.undistortPoints(imgpoints, mtx, distCoeffs, P=newcameramtx).reshape(-1,2)
            
            # draw principal point
            dst = cv2.circle(dst, (int(mtx[0, 2]), int(mtx[1, 2])), 6, (255, 0, 0), -1)

            imageio.imsave(os.path.join(undistorted_folder, os.path.basename(filenames_image)), dst)
        except:
            print("Something went wrong while undistorting the images. The distortion coefficients are probably not good. You need to take a new set of calibration images.")
            #sys.exit(0)

if __name__ == "__main__":
    
    def str2bool(v):
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Boolean value expected.')
        
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder_images", "-i", type=str, required=True)
    parser.add_argument("--output_folder", "-o", type=str, default='./output', required=False)
    parser.add_argument("--description", "-d", type=str, default="", required=False,
                        help="Optional description to add to the output file.")
    parser.add_argument("--inner_corners_height", "-ich", type=int, required=True)
    parser.add_argument("--inner_corners_width", "-icw", type=int, required=True)
    parser.add_argument("--square_sizes", "-s", type=int, default=1, required=False)
    parser.add_argument("--alpha", "-a", type=float, default=0.5, required=False,
                        help="Parameter controlling the ammount of out-of-image pixels (\"black regions\") retained in the undistorted image.")
    parser.add_argument("--threads", "-t", type=int, default=8, required=False)
    parser.add_argument("--monotonic_range", "-mr", type=float, default=-1, required=False,
                        help="Value defining the range for the distortion must be monotonic. Typical value to try 1.3. Be careful: increasing this value may negatively perturb the distortion function.")
    parser.add_argument("--debug", action="store_true", required=False)
    args = parser.parse_args()
    
    main(**vars(args))  
    
#python compute_intrinsics.py --folder_images ./frames -ich 6 -icw 8 -s 30 -t 24 --debug    