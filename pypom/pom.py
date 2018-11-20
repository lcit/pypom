#!/usr/bin/env python
""" POM related classes and functions.
"""

import os
from matplotlib.path import Path
import cv2
import numpy as np
import time
from . import utils


__author__ = "Leonardo Citraro"
__email__ = "leonardo.citraro@epfl.ch" 

X_AXIS = 0
Y_AXIS = 1
Z_AXIS = 2

class Room(object):
    """Room.
    
    Parameters
    ----------
    width : float or int
        Width of the room in world coordinate unit [m, cm, ft, inches, ...].
    height : float or int
        Height of the room in world coordinate unit [m, cm, ft, inches, ...].
    step_x : float or int
        Grid step along width in world coordinate unit [m, cm, ft, inches, ...].
        Usually between 10 cm and 30 cm.
    step_y : float or int
        Grid step along height in world coordinate unit [m, cm, ft, inches, ...].
        Usually between 10 cm and 30 cm.
    origin_x : float or int
        Origin along width of the world coordinate system w.r.t the border of the room.
    origin_y : float or int
        Origin along height of the world coordinate system w.r.t the border of the room.
    origin_z : float or int
        Origin along depth of the world coordinate system w.r.t the border of the room.
        This in general is equal to 0 unless you are doing some fancy things.
    """ 
    def __init__(self, width, height, step_x, step_y, origin_x=0, origin_y=0, origin_z=0):  
        self.width = width
        self.height = height
        self.step_x = step_x
        self.step_y = step_y
        self.origin_x = origin_x
        self.origin_y = origin_y
        self.origin_z = origin_z
        
        self.n_height = int(self.height//self.step_y)
        self.n_width = int(self.width//self.step_x)
        self.n_positions = int(self.n_height*self.n_width)
    
    def world_grid(self):
        """Compute the grid of locations in world coordinate.

        Returns
        -------
        grid : 2D numpy array (n_positions, 3)
        """
        world_grid = []
        for i in range(self.n_width):
            for j in range(self.n_height):
                world_grid.append([self.origin_x + i*self.step_x,
                                   self.origin_y + j*self.step_y,
                                   self.origin_z])   
        return np.float32(world_grid) 
    
def percentage_intersection(rectangle, image_height, image_width):
    """Compute percentage of intersection of a rectangle in an image.

    Parameters
    ----------
    rectangles : Rectangle
        The rectangle
    image_height : float or int
        Image height
    image_width : float or int
        Image width

    Returns
    -------
    Percentage of intersection : float
    """    
    image_polygon = np.array([[0, 0],
                              [0, image_height],
                              [image_width, image_height],
                              [image_width, 0]])
    img_path  = Path(image_polygon)
    
    rect_grid = np.meshgrid(np.linspace(rectangle.xmin, rectangle.xmax, 10), 
                            np.linspace(rectangle.ymin, rectangle.ymax, 10))
    rect_grid = np.vstack([rect_grid[0].ravel(), rect_grid[1].ravel()]).T
    
    return np.around(img_path.contains_points(rect_grid).sum()/len(rect_grid), 2)

def is_rect_visible(rectangle, image_height, image_width, p_visible=0.2): 
    return percentage_intersection(rectangle, image_height, image_width) > p_visible
                      
def is_rect_intersecting(rectangle, image_height, image_width):
    """Checks if rectangle is visible in an image.

    Parameters
    ----------
    rectangles : Rectangle
        The rectangle
    image_height : float or int
        Image height
    image_width : float or int
        Image width

    Returns
    -------
    True or False
    """     
    image_polygon = np.array([[0, 0],
                              [0, image_height],
                              [image_width, image_height],
                              [image_width, 0]])
    img_path  = Path(image_polygon)
    rect_path = Path(rectangle.points())

    return img_path.intersects_path(rect_path)    
    
def is_inside(rectangle, image_height, image_width):
    """Check if the whole rectangle is visible in an image.

    Parameters
    ----------
    rectangles : Rectangle
        The rectangle
    image_height : float or int
        Image height
    image_width : float or int
        Image width

    Returns
    -------
    True or False
    """  
    image_polygon = np.array([[0, 0],
                              [0, image_height-1],
                              [image_width-1, image_height-1],
                              [image_width-1, 0]])
    img_path  = Path(image_polygon)
    
    return np.alltrue(img_path.contains_points(rectangle.points()))

def constrain_rectangle_into_view(rectangle, image_height, image_width, p_visible=0.7):
    """Sets the rectangle's attribute visible according to a visibility threshold.

    Parameters
    ----------
    rectangles : Rectangle
        The rectangle
    image_height : float or int
        Image height
    image_width : float or int
        Image width
    p_visible : float (0.,1.)
        Minumum percentage of intersection required for a rectangle 
        to be considered visible in the image.

    Returns
    -------
    rectangle modifed appropriately
    """  
    if is_rect_visible(rectangle, image_height, image_width, p_visible):
        rectangle.visible = True
        rectangle.ymin = int(np.maximum(rectangle.ymin, 0))
        rectangle.ymax = int(np.minimum(rectangle.ymax, image_height-1))
        rectangle.xmin = int(np.maximum(rectangle.xmin, 0))
        rectangle.xmax = int(np.minimum(rectangle.xmax, image_width-1))
    else:
        rectangle.visible = False
        rectangle.ymin = None
        rectangle.ymax = None
        rectangle.xmin = None
        rectangle.xmax = None
    return rectangle        

class Rectangle(object):
    """Class defining a rectangle.

    Parameters
    ----------
    (ymin, xmin) : int
        Upper left corner of the rectangle
    (ymax, xmax) : int
        Bottom right corner of the rectangle
    visible : bool
        Defines whether the rectangle is to be considered visible or not

    """ 

    # (ymin, xmin) is the top-left corner of the rectangle in the image
    # (ymax, xmax) is instead the bottom-right corner of the rectangle in the image
    # the internal corner of the rectangles!
    def __init__(self, xmin=None, ymin=None, xmax=None, ymax=None, visible=None):
        self.xmin = int(xmin)
        self.ymin = int(ymin)
        self.xmax = int(xmax)
        self.ymax = int(ymax)
        self.visible = visible
    
    def __str__(self):
        return "{self.__class__.__name__}(xmin={self.xmin}, ymin={self.ymin}, xmax={self.xmax}," \
               "ymax={self.ymax}, visible={self.visible})".format(self=self)

    def points(self): 
        points = []
        points.append((self.ymin, self.xmin))
        points.append((self.ymin, self.xmax))
        points.append((self.ymax, self.xmax))
        points.append((self.ymax, self.xmin))
        return np.vstack(points) 
    
    def slices(self):
        """ Handy function to access a subarray defined by the rectangle.
        Example
        -------
        image = load_image(...)
        image[rectangle.slices()] = 0
        """
        return (slice(self.ymin, self.ymax), slice(self.xmin, self.xmax))

class Cilinder(object):
    """Class defining a cilinder. It mimics a person in a 3D space.

    Parameters
    ----------
    radius : int or float
        Radius of the cilinder. Usually between 10 and 30 cm.
    height : int or float
        Height of the cilinder. Usually between 150 and 200 cm.
    base_center : tuple (3,)
        The position of the BASE of the cilinder in 3D. 
    """ 

    # in the case you use the ground and head homographies the parameter height is no longuer meaningful
    def __init__(self, radius, height, base_center=None):
        self.radius = radius
        self.height = height
        if base_center is None:
            self.base_center = (0,0,0) # (x,y,z) meters
        else:
            self.base_center = base_center # (x,y,z) meters

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
            points.append((np.cos(a)*self.radius + self.base_center[X_AXIS], 
                           np.sin(a)*self.radius + self.base_center[Y_AXIS], 
                           self.base_center[Z_AXIS]))
        
        # the bottom and top central points of the cilinder
        points.append((self.base_center[X_AXIS], self.base_center[Y_AXIS], self.base_center[Z_AXIS]))
        return np.vstack(points)

    def top_points(self):        
        angles = np.arange(0, 2*np.pi, 0.314)
        points = []        
        for a in angles:
            points.append((np.cos(a)*self.radius + self.base_center[X_AXIS], 
                           np.sin(a)*self.radius + self.base_center[Y_AXIS], 
                           self.base_center[Z_AXIS]+self.height))
        
        # the bottom and top central points of the cilinder
        points.append((self.base_center[X_AXIS], self.base_center[Y_AXIS], self.base_center[Z_AXIS]+self.height))
        return np.vstack(points)     
        
    def project_with(self, camera):
        """Projects the cilinder into the image plane. The output is a rectangle.

        Parameters
        ----------
        camer : Camera
            Camera object defining the projection.
        """ 
        # We split bottom from top points because it possible that the full camera pose (K, R, t) is not provided.
        # The user can for example provide the ground and head homographies.
        
        image_points_top = camera.project_top_points(self.top_points())
        image_points_bot = camera.project_bottom_points(self.bottom_points())
        image_points = np.vstack([image_points_bot, image_points_top])
        
        x_proj_min = image_points[:,X_AXIS].min() # x-axis (pixels)       
        x_proj_max = image_points[:,X_AXIS].max() # x-axis (pixels) 
        
        c_proj_bot = image_points_bot[-1,Y_AXIS] # y-axis, projected bottom central point (pixels)
        c_proj_top = image_points_top[-1,Y_AXIS] # y-axis, projected top central point (pixels)
        
        # (ymin, xmin) is the top-left corner of the rectangle in the image
        # (ymax, xmax) is instead the bottom-right corner of the rectangle in the image
        ymin = c_proj_top 
        xmin = x_proj_min
        ymax = c_proj_bot
        xmax = x_proj_max        
        
        rectangle = Rectangle(xmin=xmin, ymin=ymin, xmax=xmax, ymax=ymax, visible=None)

        return rectangle
'''        
def generate_rectangles(world_grid, cameras, man_ray, man_height, view_shape, p_visible=0.7, verbose=True):
    rectangles = []
    for c, camera in enumerate(cameras):
        if verbose:
            print("[{}]::Generating rectangles..".format(camera.name))
        temp = []
        for idx, point in enumerate(world_grid):
            cilinder = Cilinder(man_ray, man_height, (point[0], point[1], 0))
            rectangle = cilinder.project_with(camera)
            rectangle = constrain_rectangle_into_view(rectangle, *view_shape, p_visible)
            temp.append(rectangle)
        rectangles.append(temp)
    return np.array(rectangles) 
'''
def generate_rectangles(world_grid, camera, man_ray, man_height, view_shape, p_visible=0.7, verbose=True):
    """Generates rectangles for a specific view.

    Parameters
    ----------
    world_grid : 2D numpy array (n_positions,3)
        World grid produced by the class Room.
    camera : Camera
        Camera object defining the projection.
    man_ray : int or float
        Radius of the cilinder/person. Usually between 10 and 30 cm.
    man_height : int or float
        Height of the cilinder/person. Usually between 150 and 200 cm. 
    view_shape : tuple (2,)
        Image shape/size (Height, Width)
    p_visible : float (0.,1.)
        Minumum percentage of intersection required for a rectangle 
        to be considered visible in the image.
    verbose : bool
        Enables status messages
    """ 
    rectangles = []
    for idx, point in enumerate(world_grid):
        cilinder = Cilinder(man_ray, man_height, (point[X_AXIS], point[Y_AXIS], point[Z_AXIS]))
        rectangle = cilinder.project_with(camera)
        rectangle = constrain_rectangle_into_view(rectangle, *view_shape, p_visible)
        rectangles.append(rectangle)
    if verbose:
        print("[{}]::Generated rectangles: {}.".format(camera.name, len(rectangles)))
    return rectangles

class DataLoader(list):
    """Class used to store the rectangles and to retrieve the images.

    Parameters
    ----------
    bg_threshold : int or float (optional)
        Threshold applied to the background subtraction images when loaded.
        This is used if the background subtraction images are grayscale instead of booleans.
    """ 
    def __init__(self, bg_threshold=128):
        self.bg_threshold = bg_threshold
    
    def add_view(self, name, rectangles, path_bgs, path_imgs=None):
        """ Adds data and paths relative to a view.

        Parameters
        ----------
        name : string
            Identifier/name for the view or the camera
        rectangles : list of Rectangle objects
            List of Rectangle objects [rect1, rect2, ...]. 
            Make sure the argument Rectangle.visible is set correctly.
        path_bgs : string
            Path to backgroud subtraction images. The path has to be "formattable"
            in order to select the images at time t.
            i.e. ".../cam0/bg/frame_{}.png"
        path_images : string (optional)
            Path to images. The path has to be "formattable"
            in order to select the images at time t.
            i.e. ".../cam0/images/frame_{}.jpg"
        """ 
        self.append({"name":name, "rectangles":rectangles, 
                     "path_bgs":path_bgs, "path_imgs":path_imgs})
    
    @property    
    def n_cams(self):
        return len(self)
    
    @property    
    def n_positions(self):
        return len(self[0]["rectangles"]) 
    
    def load_bg(self, view_idx, idx):
        image = utils.load_image(self[view_idx]["path_bgs"].format(idx))
        if image.dtype == np.dtype('bool'):
            image = np.float32(image*1.0)
        else:
            image = np.float32((image>self.bg_threshold)*1.0)                
        return image
            
    def load_bgs(self, idx):
        B = [self.load_bg(view_idx, idx) for view_idx in range(len(self))]
        return B

    def load_image(self, view_idx, idx):
        if self["path_imgs"] is None:
            return None       
        return utils.load_image(self["path_imgs"].format(idx))
    
    def load_images(self, idx):
        imgs = []
        for d in self:
            if d["path_imgs"] is not None:
                imgs.append(utils.load_image(d["path_imgs"].format(idx)))
            else:
                imgs.append(None)
        return imgs   
