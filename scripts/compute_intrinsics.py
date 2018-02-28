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

this_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(this_dir)
sys.path.append(parent_dir)
from pypom import utils

__author__ = "Leonardo Citraro"
__email__ = "leonardo.citraro@epfl.ch"

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--folder_images", "-i", type=str, required=True)
    parser.add_argument("--description", "-d", type=str, default="", required=False)
    parser.add_argument("--inner_corners_height", "-ich", type=int, required=True)
    parser.add_argument("--inner_corners_width", "-icw", type=int, required=True)
    parser.add_argument("--square_sizes", "-s", type=int, default=1, required=False)
    parser.add_argument("--threads", "-t", type=int, default=8, required=False)
    parser.add_argument("--debug", type=bool, default=False, required=False)

    args = parser.parse_args()

    folder_images=args.folder_images
    description=args.description
    inner_corners_height=args.inner_corners_height
    inner_corners_width=args.inner_corners_width
    square_sizes=args.square_sizes
    threads=args.threads
    debug=args.debug
    
    utils.rmdir("debug")
    utils.rmdir("undistorted")
         
    utils.mkdir("undistorted")
    if debug: 
        utils.mkdir("debug")            

    print("folder_images:", folder_images)
    print("description:", description)
    print("inner_corners_height:", inner_corners_height)
    print("inner_corners_width:", inner_corners_width)
    print("square_sizes:", square_sizes)
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
    
    def process_image(filename_image):
        print("Processing image {} ...".format(filename_image))
        
        gray = utils.load_image(filename_image, True)
    
        # Find the chess board corners
        ret, corners = cv2.findChessboardCorners(gray, (inner_corners_height,inner_corners_width), 
                                                 cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE)
    
        # If found, add object points, image points (after refining them)
        if ret == True:  
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            imgp = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
            
            if debug:
                gray = cv2.drawChessboardCorners(gray, (inner_corners_height,inner_corners_width), imgp, ret)
                utils.save_image(os.path.join("debug", os.path.basename(filename_image)), gray)
            
            return np.float32(imgp)
        return None
    
    pool = multiprocessing.Pool(threads)
    res = pool.map(process_image, filename_images)

    objpoints = [objp.copy() for r in res if r is not None] # 3d point in real world space
    imgpoints = [r.copy() for r in res if r is not None] # 2d points in image plane.
    
    img_shape = utils.load_image(filename_images[0], True).shape        
    
    ret, mtx, distCoeffs, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_shape[::-1], None, None)
    
    # print reprojection error
    reproj_error = 0
    for i in range(len(objpoints)):
         imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, distCoeffs)
         error = cv2.norm(imgpoints[i],imgpoints2,cv2.NORM_L2)/len(imgpoints2)
         reproj_error += error
    reproj_error /= len(objpoints)
    print("ret: {} , total reprojection error: {}".format(ret, reproj_error))
    
    d_pickle = dict({"date":current_datetime, "description":description, 
                     "K":mtx, "distCoeffs":distCoeffs, "reproj_error":reproj_error, 
                     "image_shape":img_shape})
    
    d_json = dict({"date":current_datetime, "description":description,
                   "K":mtx.tolist(), "distCoeffs":distCoeffs.tolist(),
                   "reproj_error":reproj_error, "image_shape":img_shape})
              
    utils.pickle_write(d_pickle, "intrinsics.pickle")
    utils.json_write(d_json, "intrinsics.json")  
        
    # undistorting the images    
    for filenames_image in filename_images:
    
        img = utils.load_image(filenames_image)
        h, w = img.shape[:2]
        
        try:
            newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, distCoeffs, (w, h), 0.0, (w, h), centerPrincipalPoint=True)
            
            print("Undistorting image {} -  roi={}".format(os.path.basename(filenames_image), roi))
        
            dst = cv2.undistort(img, mtx, distCoeffs, None, newcameramtx)
        
            # crop the image
            x,y,w,h = roi
            dst = dst[y:y+h, x:x+w]
        
            utils.save_image(os.path.join("undistorted", os.path.basename(filenames_image)), dst)
        except:
            print("Unable to undistort the images properly. The distortion coefficients are probably not good enough. You need to take a new set of calibration images.")
            sys.exit(0)