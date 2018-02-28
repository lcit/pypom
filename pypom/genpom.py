#!/usr/bin/env python
""" Classes ad functions used to generate a POM (Probabilistic Occupancy Map).
"""

import os
from matplotlib.path import Path
import cv2
import numpy as np
import xml.etree.ElementTree
import numbers
from . import utils

__author__ = "Leonardo Citraro"
__email__ = "leonardo.citraro@epfl.ch"

def mkdir(directory):
    directory = os.path.abspath(directory)
    if not os.path.exists(directory):
        os.makedirs(directory)
        
def transform_points(H, points):
    # convertPointsFromHomogeneous has some problem when dealing with non float values!
    points = points.astype(np.float)
    
    # if points is just a vector we need to add a new dimension to make it working correctly
    if np.ndim(points) == 1:
        points = points[np.newaxis,:]
    return cv2.convertPointsFromHomogeneous(np.dot(H, cv2.convertPointsToHomogeneous(points)[:,0,:].T).T)[:,0,:] 

def project_KRt(points, R, t, K, dist=None):
    homogeneous = np.dot(K, (np.dot(R, points.T) + t)).T
    image_points = homogeneous[:,:2] / homogeneous[:,[2]]  
    if dist is not None:
        image_points = cv2.undistortPoints(image_points[:,np.newaxis].astype(np.float32), K, dist) 
    return image_points          
        
class Camera(object):
    def __init__(self, K=None, R=None, t=None, distCoeffs=None, rvec=None, H_ground=None, H_head=None, head_height=None):
        self.K = K
        self.R = R
        self.t = t
        self.distCoeffs = distCoeffs
        self.rvec = rvec
        self.H_ground = H_ground
        self.H_head = H_head
        self.head_height = head_height
        
        assert(isinstance(self.K, np.ndarray) or self.K is None)
        assert(isinstance(self.R, np.ndarray) or self.R is None)
        assert(isinstance(self.t, np.ndarray) or self.t is None)
        assert(isinstance(self.distCoeffs, np.ndarray) or self.distCoeffs is None)
        assert(isinstance(self.rvec, np.ndarray) or self.rvec is None)
        assert(isinstance(self.H_ground, np.ndarray) or self.H_ground is None)
        assert(isinstance(self.H_head, np.ndarray) or self.H_head is None)
        assert(isinstance(self.head_height, int) or self.head_height is None)
        
    def project_points(self, world_points):
        if self.rvec is not None:
            image_points, _ = cv2.projectPoints(world_points, self.rvec, self.tvec, self.K, self.distCoeffs)
            return image_points.squeeze() 
        elif self.R is not None:
            return project_KRt(world_points, self.R, self.t, self.K, self.distCoeffs)        
        else:
            raise RuntimeError("Neither rvec nor R are defined!")
    
    def project_bottom_points(self, world_points):
        if self.H_ground is not None:
            if isinstance(self.H_ground, numbers.Number):
                return 
            else:
                return transform_points(self.H_ground, world_points[:,:2])
        elif self.rvec is not None or self.R is not None:
            return self.project_points(world_points)       
        else:
            raise RuntimeError("Neither rvec nor H_ground are defined!")
    
    def project_top_points(self, world_points):
        if self.H_head is not None:
            return transform_points(self.H_head, world_points[:,:2])
        elif self.head_height is not None:
            # special case, doing so we are telling the caller of this function 
            # to use the bottom points as only reference to build the rectangle
            return self.head_height 
        elif self.rvec is not None or self.R is not None:
            return self.project_points(world_points)        
        else:
            raise RuntimeError("Neither rvec nor H_head are defined!")   

    @classmethod
    def from_json(cls, intrinsics_json, extrinsics_json, downsampling):
        K, dist, image_shape_i = utils.retrieve_intrinsics_from_json(intrinsics_json)        
        R, t, image_shape_e, unit = utils.retrieve_extrinsics_from_json(extrinsics_json)
        if image_shape_i[0]!=image_shape_e[0] or image_shape_i[1]!=image_shape_e[1]:
            raise ValueError("Image shapes inside intrinsics and extrinsics files are not equal!")
            
        K = utils.scale_homography(K, 1.0/downsampling)
        
        return cls(K=K, R=R, t=t)
                             

class Rectangle(object):
    def __init__(self, cam, idx, xmin, ymin, xmax, ymax):
        self.cam = cam
        self.idx = idx
        self.xmin = int(xmin)
        self.ymin = int(ymin)
        self.xmax = int(xmax)
        self.ymax = int(ymax)
    
    def __str__(self):
        return "{self.__class__.__name__}(cam={self.cam}, idx={self.idx}, xmin={self.xmin}," \
               "ymin={self.ymin}, xmax={self.xmax}, ymax={self.ymax})".format(self=self)

    def points(self): 
        points = []
        points.append((self.xmin, self.ymin))
        points.append((self.xmin, self.ymax))
        points.append((self.xmax, self.ymax))
        points.append((self.xmax, self.ymin))
        return np.vstack(points)
    
    def is_intersecting(self, image_width, image_height):
        
        image_polygon = np.array([[0, 0],
                                  [0, image_height],
                                  [image_width, image_height],
                                  [image_width, 0]])
        img_path  = Path(image_polygon)
        rect_path = Path(self.points())

        return img_path.intersects_path(rect_path)    
    
    def percentage_intersection(self, image_width, image_height):
        
        image_polygon = np.array([[0, 0],
                                  [0, image_height],
                                  [image_width, image_height],
                                  [image_width, 0]])
        img_path  = Path(image_polygon)
        
        rect_grid = np.meshgrid(np.linspace(self.xmin, self.xmax, 10), 
                                np.linspace(self.ymin, self.ymax, 10))
        rect_grid = np.vstack([rect_grid[0].ravel(), rect_grid[1].ravel()]).T
        
        return img_path.contains_points(rect_grid).sum()/len(rect_grid)
    
    def is_visible(self, image_width, image_height, p_visible):
        return self.percentage_intersection(image_width, image_height) > p_visible
    
    def is_inside(self, image_width, image_height):
        image_polygon = np.array([[0, 0],
                                  [0, image_height-1],
                                  [image_width-1, image_height-1],
                                  [image_width-1, 0]])
        img_path  = Path(image_polygon)
        
        return np.alltrue(img_path.contains_points(self.points()))
    
    def to_string(self, image_width, image_height, p_visible):     
        if self.is_inside(image_width, image_height) and self.is_visible(image_width, image_height, p_visible):
            return "RECTANGLE {} {} {} {} {} {}".format(self.cam, self.idx, 
                                                       self.xmin, self.ymin, self.xmax, self.ymax)
        else:
            return "RECTANGLE {} {} notvisible".format(self.cam, self.idx)
        
    @classmethod
    def from_string(cls, string):
        return cls(*[int(x) for x in string.split(' ')])   

class Cilinder(object):
    # in the case you use the ground and head homographies the parameter height is no longuer meaningful
    def __init__(self, radius, height, base_center=None):
        self.radius = radius
        self.height = height
        if base_center is None:
            self.base_center = (0, 0) # (x,y) meters
        else:
            self.base_center = base_center # (x,y) meters
            
    def __str__(self):
        return "{self.__class__.__name__}(radius={self.radius}, height={self.height}," \
               "base_center={self.base_center})".format(self=self)            
        
    def points(self):  
        return np.vstack([self.bottom_points()[:-1], self.top_points()[:-1], 
                          self.bottom_points()[-1], self.top_points()[-1]]) 
    
    def bottom_points(self):        
        angles = np.arange(0, 2*np.pi, 0.314)
        points = []        
        for a in angles:
            points.append((np.cos(a)*self.radius + self.base_center[0], 
                           np.sin(a)*self.radius + self.base_center[1], 
                           self.base_center[2]))
        
        # the bottom and top central points of the cilinder
        points.append((self.base_center[0], self.base_center[1], self.base_center[2]))
        return np.vstack(points) 

    def top_points(self):        
        angles = np.arange(0, 2*np.pi, 0.314)
        points = []        
        for a in angles:
            points.append((np.cos(a)*self.radius + self.base_center[0], 
                           np.sin(a)*self.radius + self.base_center[1], 
                           self.base_center[2]+self.height))
        
        # the bottom and top central points of the cilinder
        points.append((self.base_center[0], self.base_center[1], self.base_center[2]+self.height))
        return np.vstack(points)     
        
    def project(self, camera):
        
        # we split bottom from top points because it possible that the full camera pose is not provided.
        # Instead you can provide ground and head homographies.
        
        image_points_bot = camera.project_bottom_points(self.bottom_points())
        image_points_top = camera.project_top_points(self.top_points())
        if isinstance(image_points_top, int):
            # special case, image_points_top is an integer that defines the height(top part) of the rectangle
            image_points = np.vstack([image_points_bot])
        else:
            image_points = np.vstack([image_points_bot, image_points_top])
        
        x_proj_min = image_points[:,0].min() # (pixels)       
        x_proj_max = image_points[:,0].max() # (pixels) 
        
        c_proj_bot = image_points_bot[-1,1] # projected bottom central point (pixels)
        if isinstance(image_points_top, int):
            c_proj_top = image_points_top
        else:
            c_proj_top = image_points_top[-1,1] # projected top central point (pixels)
        
        xmin = x_proj_min
        ymax = c_proj_bot
        xmax = x_proj_max
        ymin = c_proj_top

        return xmin, ymin, xmax, ymax
    
class POM(object):
    def __init__(self, img_width, img_height, rectangles, 
                 input_view_format="./images/cam_%c/img_%f.png", 
                 result_view_format="./results/result-f%f-c%c.png",
                 result_format="./results/proba-f%f.dat", 
                 convergence_view_format="./results/convergence/f%f-c%c-i%i.png", 
                 prior=0.01, sigma_image_density=0.01, max_nb_solver_iterations=100, 
                 proba_ignored=1.0, idx_start=0, process=10, p_rect_visible=0.2):        
        self.img_width = img_width
        self.img_height = img_height
        self.n_cams = len(rectangles)
        self.n_poss = len(rectangles[0])
        
        # rectangles is a list of list where the first index is the camera
        self.rectangles = rectangles

        if input_view_format is None:
            raise ValueError("input_view_format cannot be None!")
        self.input_view_format = input_view_format
        self._check_png(self.input_view_format)
        mkdir(os.path.dirname(self.input_view_format))        
        
        self.result_view_format = result_view_format
        self._check_png(self.result_view_format)
        mkdir(os.path.dirname(self.result_view_format)) 
           
        self.result_format = result_format
        mkdir(os.path.dirname(self.result_format))
            
        self.convergence_view_format = convergence_view_format  
        self._check_png(self.convergence_view_format)
        mkdir(os.path.dirname(self.convergence_view_format))
        
        self.prior = prior
        self.sigma_image_density = sigma_image_density        
        self.max_nb_solver_iterations = max_nb_solver_iterations
        self.proba_ignored = proba_ignored
        
        self.idx_start = idx_start
        self.process = process
        
        self.p_rect_visible = p_rect_visible
        
    def _check_png(self, filename):
        assert(filename.lower().split('.')[-1] == 'png')

    def write_to_file(self, filename):
        mkdir(os.path.dirname(filename))
        
        text_file = open(filename, "w")
        text_file.write("ROOM {} {} {} {}\n\n".format(self.img_width, self.img_height, self.n_cams, self.n_poss))
        for c in range(self.n_cams):
            for i in range(self.n_poss):
                rect = self.rectangles[c][i]
                text_file.write(rect.to_string(self.img_width, self.img_height, self.p_rect_visible)+"\n")       
        text_file.write("\n")
        
        text_file.write("INPUT_VIEW_FORMAT {}\n\n".format(self.input_view_format))
        if self.result_view_format is not None:
            text_file.write("RESULT_VIEW_FORMAT {}\n\n".format(self.result_view_format))  
        if self.result_format is not None:
            text_file.write("RESULT_FORMAT {}\n\n".format(self.result_format))             
        if self.convergence_view_format is not None:
            text_file.write("CONVERGENCE_VIEW_FORMAT {}\n\n".format(self.convergence_view_format))    
        if self.prior is not None:
            text_file.write("PRIOR {}\n".format(self.prior))   
        if self.sigma_image_density is not None:
            text_file.write("SIGMA_IMAGE_DENSITY {}\n\n".format(self.sigma_image_density)) 
        if self.max_nb_solver_iterations is not None:
            text_file.write("MAX_NB_SOLVER_ITERATIONS {}\n\n".format(self.max_nb_solver_iterations)) 
        if self.proba_ignored is not None:
            text_file.write("PROBA_IGNORED {}\n\n".format(self.proba_ignored)) 
        if self.idx_start is not None and self.process is not None:
            text_file.write("PROCESS {} {}\n\n".format(self.idx_start, self.process))             
        
        text_file.close()
        
def generate_rectangles(world_grid, cameras, man_ray, man_height, verbose=True):
    rectangles = []
    for c, camera in enumerate(cameras):
        if verbose:
            print("Processing rectangles for camera {}..".format(c))
        temp = []
        for idx, point in enumerate(world_grid):
            cilinder = Cilinder(man_ray, man_height, (point[0], point[1], 0))
            rect = Rectangle(c, idx, *cilinder.project(camera)) 
            temp.append(rect)
        rectangles.append(temp)
    return rectangles        
