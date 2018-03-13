#!/usr/bin/env python
""" Camera.
"""

import os
from matplotlib.path import Path
import cv2
import numpy as np
import numbers
from . import utils

__author__ = "Leonardo Citraro"
__email__ = "leonardo.citraro@epfl.ch" 

def retrieve_intrinsics_from_json(filename):
    calibration = utils.json_read(filename)

    K = utils.search_in_dict(calibration, "K", [("mtx"),("proj")])
    if K is None:
        raise RuntimeError("Unable to retrieve 'K' from file {}".format(filename))
        
    dist = utils.search_in_dict(calibration, "distCoeffs", [("dist")])      
    if dist is None:
        raise RuntimeError("Unable to retrieve 'distCoeffs' from file {}".format(filename)) 
        
    image_shape = utils.search_in_dict(calibration, "image_shape", [("im","shape"), ("im","size"), ("size"), ("shape")])
    if image_shape is None:
        raise RuntimeError("Unable to retrieve 'image_shape' from file {}".format(filename))        

    return np.float64(K).reshape(3,3), np.float64(dist).ravel(), np.float64(image_shape).ravel()

def retrieve_extrinsics_from_json(filename):
    calibration = utils.json_read(filename)
    
    R = utils.search_in_dict(calibration, "R", [("Rot", "rvec")])
    if R is None:
        raise RuntimeError("Unable to retrieve 'R' from file {}".format(filename))
    if len(np.array(R).ravel()) == 3:
        # in the case the rotation is encoded as Rodrigues vector (rvec)
        R = utils.rotvector_to_rotmatrix(np.float64(R))
    else:
        R = np.float64(R).reshape(3,3)
        
    t = utils.search_in_dict(calibration, "t", [("trans")])
    if t is None:
        raise RuntimeError("Unable to retrieve 't' from file {}".format(filename))

    image_shape = utils.search_in_dict(calibration, "image_shape", [("im","shape"), ("im","size"), ("size"), ("shape")])
    if image_shape is None:
        raise RuntimeError("Unable to retrieve 'image_shape' from file {}".format(filename))
        
    unit = utils.search_in_dict(calibration, "unit", [("unit")])
    if unit is None:
        raise RuntimeError("Unable to retrieve 'unit' from file {}".format(filename))                

    return R, np.float64(t).reshape(3,1), np.float64(image_shape).ravel(), unit

def retrieve_homographies_from_json(filename):
    homographies = utils.json_read(filename)
    
    Hbottom = utils.search_in_dict(homographies, "Hbottom", [("bot")])
    if Hbottom is None:
        raise RuntimeError("Unable to retrieve 'Hbottom' from file {}".format(filename))
        
    Htop_or_height = utils.search_in_dict(homographies, "Htop", [("top"), ("height"), ("head")])
    if Htop_or_height is None:
        raise RuntimeError("Unable to retrieve 'Htop or head-height' from file {}".format(filename))
    if len(np.array(Htop_or_height).ravel()) != 9:
        # in the case the top homography is replaced with just an height
        # the height must be in percentage from the top
        Htop_or_height = float(Htop_or_height)
    else:
        Htop_or_height = np.float64(Htop_or_height).reshape(3,3)    

    image_shape = utils.search_in_dict(homographies, "image_shape", [("im","shape"), ("im","size"), ("size"), ("shape")])
    if image_shape is None:
        raise RuntimeError("Unable to retrieve 'image_shape' from file {}".format(filename))
        
    unit = utils.search_in_dict(homographies, "unit", [("unit")])
    if unit is None:
        raise RuntimeError("Unable to retrieve 'unit' from file {}".format(filename))                

    return np.float64(Hbottom).reshape(3,3), Htop_or_height, np.float64(image_shape).ravel(), unit

def retrieve_image_and_world_points_from_json(filename):
    points = utils.json_read(filename)
    
    image_points = utils.search_in_dict(points, "image_points", [("image","point")])
    if image_points is None:
        raise RuntimeError("Unable to retrieve 'image_points' from file {}".format(filename))
        
    world_points = utils.search_in_dict(points, "world_points", [("world","point"), ("model","point")])
    if world_points is None:
        raise RuntimeError("Unable to retrieve 'world_points' from file {}".format(filename))
        
    image_shape = utils.search_in_dict(points, "image_shape", [("im","shape"), ("im","size"), ("size"), ("shape")])
    if image_shape is None:
        raise RuntimeError("Unable to retrieve 'image_shape' from file {}".format(filename))
        
    unit = utils.search_in_dict(points, "unit", [("unit")])
    if unit is None:
        raise RuntimeError("Unable to retrieve 'unit' from file {}".format(filename))
        
    undistorted = utils.search_in_dict(points, "distCoeffs", [("undist")])
    if undistorted is None:
        raise RuntimeError("Unable to retrieve 'undistorted' from file {}".format(filename))
        
    return np.float64(image_points), np.float64(world_points), np.float64(image_shape).ravel(), unit, undistorted

class CameraKRt(object):
    def __init__(self, name, K, R, t, scale=1): 
        self.name = name
        self.K = K
        self.R = R
        self.t = t
        self.scale = scale
        
    def project_points(self, world_points):
        return utils.project_KRt(world_points, self.R, self.t, self.K)*self.scale      
    
    def project_bottom_points(self, world_points):
        return self.project_points(world_points)       
    
    def project_top_points(self, world_points):
        return self.project_points(world_points)        

    @classmethod
    def from_json(cls, name, intrinsics_json, extrinsics_json, view_shape, unit="m", invert=False):
        K, dist, image_shape_i = retrieve_intrinsics_from_json(intrinsics_json)        
        R, t, image_shape_e, unit_e = retrieve_extrinsics_from_json(extrinsics_json)
        if not np.all(image_shape_i == image_shape_e):
            raise ValueError("The intrinsics and extrinsics parameters must be computed on images of equal size!")
            
        if invert:
            R, t = utils.invert_Rt(R, t)
        
        # we convert the unit of the extrinsics, R remain the same t changes
        if unit_e != unit:
            t = utils.value_unit_conversion(t, in_unit=unit_e, out_unit=unit)
            print("[{}]::Extrinsics converted from '{}' to '{}'.".format(name, unit_e, unit))
        else:
            print("[{}]::Extrinsics kept in '{}'.".format(name, unit_e))
            
        # we compute the scale between the output image shape and the shape used for the calibration
        scale1 = view_shape[0]/image_shape_i[0]
        scale2 = view_shape[1]/image_shape_i[1]
        if scale1 != scale2:
            raise ValueError("View shape must be a multiple of the size of the images used for the intrinsics and extrinsics! "\
                             "It could be for example ({},{}) or ({},{}) etc..".format(image_shape_i[0]/2.0, image_shape_i[1]/2.0,
                             image_shape_i[0]/3.0, image_shape_i[1]/3.0)) 
        
        return cls(name=name, K=K, R=R, t=t, scale=scale1)

class CameraHbotHtop(object):
    def __init__(self, name, Hbottom, Htop, scale=1):
        self.name = name
        self.Hbottom = Hbottom
        self.Htop = Htop
        self.scale = scale
        
    def project_points(self, world_points):
        raise NotImplementedError()  
    
    def project_bottom_points(self, world_points):
        return utils.transform_points(self.Hbottom, world_points[:,:2])*self.scale
    
    def project_top_points(self, world_points):
        return utils.transform_points(self.Htop, world_points[:,:2])*self.scale

    @classmethod
    def from_json(cls, name, homographies_json, view_shape, unit="m", invert=False):
        Hbottom, Htop, image_shape_h, unit_h = retrieve_homographies_from_json(homographies_json) 
        
        if invert:
            Hbottom = np.linalg.inv(Hbottom)
            Htop = np.linalg.inv(Htop)

        scale = utils.unit_conversion[unit_h][unit](1)
        Hbottom = utils.scale_homography_right(Hbottom, scale)
        Htop = utils.scale_homography_right(Htop, scale)
            
        # we must convert the intrinsics as well according to the image size
        scale1 = view_shape[0]/image_shape_h[0]
        scale2 = view_shape[1]/image_shape_h[1]
        if scale1 != scale2:
            raise ValueError("View shape must be a multiple of the size of the images used for computing the homographies! "\
                             "It could be for example ({},{}) or ({},{}) etc..".format(image_shape_h[0]/2.0, image_shape_h[1]/2.0,
                             image_shape_h[0]/3.0, image_shape_h[1]/3.0)) 
        
        return cls(name=name, Hbottom=Hbottom, Htop=Htop, scale=scale1)    

class CameraHbotHeight(object):
    # head_height is in pixels from the top of the image
    # p_head_height is a percentage 
    def __init__(self, Hbottom, head_height, scale=1):
        self.Hbottom = Hbottom
        self.head_height = head_height
        self.scale = scale
        
    def project_points(self, world_points):
        raise NotImplementedError()     
    
    def project_bottom_points(self, world_points):
        return utils.transform_points(self.Hbottom, world_points[:,:2])*self.scale
    
    def project_top_points(self, world_points):
        points = self.project_bottom_points(world_points)
        return np.hstack([np.ones((points.shape[0],1))*self.head_height, points[:,[1]]])

    @classmethod
    def from_json(cls, name, homographies_json, view_shape, unit="m", invert=False):
        Hbottom, p_head_height, image_shape_h, unit_h = retrieve_homographies_from_json(homographies_json) 
        
        if invert:
            Hbottom = np.linalg.inv(Hbottom)

        scale = utils.unit_conversion[unit_h][unit](1)
        Hbottom = utils.scale_homography_right(Hbottom, scale)
                    
        # we must convert the intrinsics as well according to the image size
        scale1 = view_shape[0]/image_shape_h[0]
        scale2 = view_shape[1]/image_shape_h[1]
        if scale1 != scale2:
            raise ValueError("View shape must be a multiple of the size of the images used for computing the homographies! "\
                             "It could be for example ({},{}) or ({},{}) etc..".format(image_shape_h[0]/2.0, image_shape_h[1]/2.0,
                             image_shape_h[0]/3.0, image_shape_h[1]/3.0))
            
        head_height = int(view_shape[0]*p_head_height)
        
        return cls(name=name, Hbottom=Hbottom, head_height=head_height, scale=scale1)