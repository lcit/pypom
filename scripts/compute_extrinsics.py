#!/usr/bin/env python3
import cv2
import argparse
import os
import sys
import numpy as np
import inspect
import datetime

this_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(this_dir)
sys.path.append(parent_dir)
from pypom import utils

def main(points_json="points.json",
         intrinsics_json="intrinsics.json",
         description=""):
    """ Extraction of a sequence of frame from a video file.

    This script require Opencv with FFmpeg support.
    
    Parameters
    ----------
    points_json: str
        filename of a JSON file containing image points, world points and the 
        shape of an undistorted image
    intrinsics_json: str
        filename of a JSON file containing the intrinsics paramters
    """

    # the image points are extracted from an undistorted image!
    image_points, world_points, image_shape, unit, undistorted = utils.retrieve_image_and_world_points_from_json(points_json)
    
    assert(image_points is not None)
    assert(world_points is not None)
    assert(image_shape is not None)
    assert(unit is not None and isinstance(unit, str))
    assert(undistorted is not None and isinstance(undistorted, bool))
    
    K, dist, _ = utils.retrieve_intrinsics_from_json(intrinsics_json)
    
    if not undistorted:
        image_points = cv2.undistortPoints(image_points[:,np.newaxis].astype(np.float32), K, dist)

    Hr = cv2.findHomography(world_points, image_points)[0]

    print("----- Inputs -----")
    print("Hr\n", Hr)
    print("K\n", K)
    print("world_points\n", world_points)
    print("image_shape\n", image_shape)

    R, t = utils.homography_to_transformation(Hr, K, world_points, image_shape)

    print("----- Ouputs -----")
    print("R\n", R)
    print("t\n", t)
    
    current_datetime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    d_pickle = dict({"date":current_datetime, "description":description, 
                    "R":R, "t":t, "Hr":Hr, "image_shape":image_shape, "unit":unit})
    
    d_json = dict({"date":current_datetime, "description":description, 
                   "R":R.tolist(), "t":t.tolist(), "Hr":Hr.tolist(), "image_shape":image_shape.tolist(), "unit":unit})
              
    utils.pickle_write(d_pickle, "extrinsics.pickle")
    utils.json_write(d_json, "extrinsics.json")

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--points_json", "-p", type=str, required=True)
    parser.add_argument("--intrinsics_json", "-i", type=str, required=True)
    parser.add_argument("--description", "-d", type=str, default="", required=False)

    args = parser.parse_args()
    main(**vars(args))    
