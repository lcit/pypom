#!/usr/bin/env python
""" Classes ad functions used to generate a POM (Probabilistic Occupancy Map).
"""

import os
from matplotlib.path import Path
import cv2
import numpy as np
import time
import utils
import core

__author__ = "Leonardo Citraro"
__email__ = "leonardo.citraro@epfl.ch" 

class Room(object):
    def __init__(self, width, height, step_x, step_y, origin_x, origin_y, n_cams):  
        self.width = width
        self.height = height
        self.step_x = step_x
        self.step_y = step_y
        self.origin_x = origin_x
        self.origin_y = origin_y
        self.n_cams = n_cams
        
        self.n_height = self.height//self.step_y
        self.n_width = self.width//self.step_x
        self.n_positions = self.n_height*self.n_width
    
    def world_grid(self):
        world_grid = []
        for i in range(self.n_width):
            for j in range(self.n_height):
                world_grid.append([self.origin_x - i*self.step_x,
                                   self.origin_y - j*self.step_y,
                                   0])   
        return np.float64(world_grid) 
    
def percentage_intersection(rectangle, image_height, image_width):
    
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
    
    image_polygon = np.array([[0, 0],
                              [0, image_height],
                              [image_width, image_height],
                              [image_width, 0]])
    img_path  = Path(image_polygon)
    rect_path = Path(rectangle.points())

    return img_path.intersects_path(rect_path)    
    
def is_inside(rectangle, image_height, image_width):
    image_polygon = np.array([[0, 0],
                              [0, image_height-1],
                              [image_width-1, image_height-1],
                              [image_width-1, 0]])
    img_path  = Path(image_polygon)
    
    return np.alltrue(img_path.contains_points(rectangle.points()))

def constrain_rectangle_into_view(rectangle, image_height, image_width, p_visible=0.7):
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
        return (slice(self.ymin, self.ymax), slice(self.xmin, self.xmax))

class Cilinder(object):
    # in the case you use the ground and head homographies the parameter height is no longuer meaningful
    def __init__(self, radius, height, base_center=None):
        self.radius = radius
        self.height = height
        if base_center is None:
            self.base_center = (0,0) # (x,y) meters
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
                           0))
        
        # the bottom and top central points of the cilinder
        points.append((self.base_center[0], self.base_center[1], 0))
        return np.vstack(points)

    def top_points(self):        
        angles = np.arange(0, 2*np.pi, 0.314)
        points = []        
        for a in angles:
            points.append((np.cos(a)*self.radius + self.base_center[0], 
                           np.sin(a)*self.radius + self.base_center[1], 
                           self.height))
        
        # the bottom and top central points of the cilinder
        points.append((self.base_center[0], self.base_center[1], self.height))
        return np.vstack(points)     
        
    def project_with(self, camera):        
        # we split bottom from top points because it possible that the full camera pose is not provided.
        # Instead you can provide ground and head homographies.
        
        image_points_top = camera.project_top_points(self.top_points())
        image_points_bot = camera.project_bottom_points(self.bottom_points())
        image_points = np.vstack([image_points_bot, image_points_top])
        
        x_proj_min = image_points[:,1].min() # (pixels)       
        x_proj_max = image_points[:,1].max() # (pixels) 
        
        c_proj_bot = image_points_bot[-1,0] # projected bottom central point (pixels)
        c_proj_top = image_points_top[-1,0] # projected top central point (pixels)
        
        # (ymin, xmin) is the top-left corner of the rectangle in the image
        # (ymax, xmax) is instead the bottom-right corner of the rectangle in the image
        ymin = c_proj_top 
        xmin = x_proj_min
        ymax = c_proj_bot
        xmax = x_proj_max        
        
        rectangle = Rectangle(xmin=xmin, ymin=ymin, xmax=xmax, ymax=ymax, visible=None)

        return rectangle
        
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

class Solver(object):
    def __init__(self, rectangles, prior=0.001, max_iter=100, sigma=0.01, step=0.9, eps=1e-12):
        self.rectangles = rectangles
        self.prior = prior
        self.max_iter = max_iter
        self.sigma = sigma
        self.step = step
        self.eps = eps
        
        self.n_cams = rectangles.shape[0]
        self.n_positions = rectangles.shape[1]
        
        self.type_t = np.float64
        
        e = np.full(self.n_positions, self.prior, self.type_t)
        self.lambd = np.log((1-e)/e)        
        
        # preparing the rectangles for the cpp wrapper
        self._rectangles = []
        for c in range(self.n_cams):
            temp = []
            for k, r in enumerate(rectangles[c]):
                if r.visible:
                    temp += [r.ymin,r.ymax,r.xmin,r.xmax]
                else:
                    temp += [-1,-1,-1,-1]
            self._rectangles.append(np.int32(temp)) 
        self._rectangles = np.array(self._rectangles)
        
    def run(self, B, q=None, verbose=True, debug=False):
        start = time.time()
        
        if q is None:
            q = np.full(self.n_positions, self.prior, self.type_t)
        
        view_shape = B[0].shape[:2]
        n_stab = 0
        qs = []
        
        if debug:
            for c in range(self.n_cams):
                utils.rmdir("./c{}/A".format(c))
                utils.mkdir("./c{}/A".format(c))   
        
        for i in range(self.max_iter):  
             
            utils.loading_bar(i, msg="Iteration:{}".format(i), max=self.max_iter-1)
            
            A = np.ones((self.n_cams,)+view_shape, self.type_t)
            Ai = np.ones((self.n_cams,)+view_shape, self.type_t)
            BAi = np.ones((self.n_cams,)+view_shape, self.type_t)
            
            for c in range(self.n_cams):
 
                A_ = core.compute_A_(view_shape, self._rectangles[c], q)
                             
                A[c] = 1-A_ # 1-xk(1-qkAk)
                Ai[c] = core.integral_image(A_) # (1-A)
                BAi[c] = core.integral_image(A_*B[c]) # Bx(1-A)
    
                if debug:
                    utils.save_image("./c{}/A/A{}.JPG".format(c,i), np.uint8(255*np.dstack([A[c], B[c], B[c]])))
                
            lAl = [np.sum(A[c]) for c in range(self.n_cams)]
            lBxAl = [np.sum(B[c]*A[c]) for c in range(self.n_cams)]
            lBl = [np.sum(B[c]) for c in range(self.n_cams)]
    
            psi0 = np.zeros((self.n_cams, self.n_positions), self.type_t)
            psi1 = np.zeros((self.n_cams, self.n_positions), self.type_t) 
             
            for c in range(self.n_cams):
                (p0,p1) = core.compute_psi(Ai[c], BAi[c], lAl[c], lBxAl[c], lBl[c], self._rectangles[c], q) 
                psi0[c] = p0  
                psi1[c] = p1          
    
            psi1_psi0 = np.minimum(1/self.sigma*(psi1 - psi0).sum(0), 30)
            q_new = np.array(q*self.step + (1-self.step)/(1+np.exp(self.lambd + psi1_psi0)), self.type_t)
            qs.append(q_new)
            
            diff = np.abs((q_new-q)).mean()
            if diff  < 1e-6:
                n_stab += 1
                if n_stab > 5:
                    if verbose:                        
                        utils.loading_bar(i, msg="[{:0.2f}s]Solved at iteration: {}".format(time.time()-start, i), max=self.max_iter-1)
                        print("------------------------")
                    return qs
            else:
                n_stab = 0        
            q = q_new
                
        utils.loading_bar(i, msg="[{:0.2f}s]Max iteration reached: {}".format(time.time()-start, i), max=self.max_iter-1)
        print("------------------------")
        return qs       

#def integral_array(a):
#    S = a
#    for i in range(a.ndim):
#        S = S.cumsum(axis=i)
#    return S
#
#def integral_sum(a):
#    return a[-1,-1]+a[0,0]-a[-1,0]-a[0,-1]
#        
#def run(B, rectangles, q, lambd, prior=0.001, iterations=100, sigma=0.01, step=0.8, eps=1e-12, debug=False):
#    
#    n_stab = 0
#    n_positions = room.n_positions
#    n_cams = len(rectangles)
#    print(n_cams, n_positions)
#    H = B[0].shape[0]
#    W = B[0].shape[1]
#    
#    
#    if debug:
#        for c in range(n_cams):
#            utils.rmdir("./c{}/A".format(c))
#            utils.mkdir("./c{}/A".format(c))  
#    
#    for i in range(iterations):  
#        
#        if i%10==0:
#            print("Iteration: ",i)
#        
#        A = np.ones((n_cams,)+view_shape, type_t)
#        Ai = np.ones((n_cams,)+view_shape, type_t)
#        BAi = np.ones((n_cams,)+view_shape, type_t)
#        
#        for c, rs in enumerate(rectangles):
#            '''
#            A_ = np.ones(view_shape, type_t)
#            for k in range(n_positions):   
#                #if r.visible:
#                ymin = rs[(k*4)+0]
#                ymax = rs[(k*4)+1]
#                xmin = rs[(k*4)+2]
#                xmax = rs[(k*4)+3]
#                if ymin != -1:                    
#                    ymin = np.maximum(ymin, 0)
#                    ymax = np.minimum(ymax, H)
#                    xmin = np.maximum(xmin, 0)
#                    xmax = np.minimum(xmax, W)
#                    A_[ymin:ymax, xmin:xmax] *= 1-q[k]
#            
#            '''
#            A_ = pom_core.compute_A_(view_shape, rs, q)
#             
#            
#            A[c] = 1-A_ # 1-xk(1-qkAk)
#            Ai[c] = pom_core.integral_image(A_) # (1-A)
#            BAi[c] = pom_core.integral_image(A_*B[c]) # Bx(1-A)
#            '''
#            
#            A[c] = 1-A_ # 1-xk(1-qkAk)
#            Ai[c] = integral_array(A_) # (1-A)
#            BAi[c] = integral_array(B[c]*A_) # Bx(1-A)
#            '''
#            if debug:
#                utils.save_image("./c{}/A/A{}.JPG".format(c,i), np.dstack([A[c], B[c], B[c]]))
#            
#        lAl = [np.sum(A[c]) for c in range(n_cams)]
#        lBxAl = [np.sum(B[c]*A[c]) for c in range(n_cams)]
#        lBl = [np.sum(B[c]) for c in range(n_cams)]
#
#        psi0 = np.zeros((n_cams, n_positions), type_t)
#        psi1 = np.zeros((n_cams, n_positions), type_t) 
#         
#        for c, rs in enumerate(rectangles):
#            (p0,p1) = pom_core.compute_psi(Ai[c], BAi[c], lAl[c], lBxAl[c], lBl[c], rs, q) 
#            psi0[c] = p0  
#            psi1[c] = p1          
#        '''
#        for c,rs in enumerate(rectangles):
#            for k in range(n_positions):   
#                #if r.visible:
#                ymin = rs[(k*4)+0]
#                ymax = rs[(k*4)+1]
#                xmin = rs[(k*4)+2]
#                xmax = rs[(k*4)+3]
#                if ymin != -1:                    
#                    ymin = np.maximum(ymin, 0)
#                    ymax = np.minimum(ymax, H)
#                    xmin = np.maximum(xmin, 0)
#                    xmax = np.minimum(xmax, W)
#                    
#                    #l1_AxAkl = pom_core.integral_sum(Ai[c], r.ymin, r.ymax, r.xmin, r.xmax)
#                    l1_AxAkl = integral_sum(Ai[c][ymin:ymax,xmin:xmax])
#                    
#                    lAk0l = lAl[c] - q[k]/(1-q[k])*l1_AxAkl
#                    lAk1l = lAl[c] +               l1_AxAkl
#                    
#                    #lBx1_AxAkl = pom_core.integral_sum(BAi[c], r.ymin, r.ymax, r.xmin, r.xmax)
#                    lBx1_AxAkl = integral_sum(BAi[c][ymin:ymax, xmin:xmax])
#                    lBxAk0l = lBxAl[c] - q[k]/(1-q[k])*lBx1_AxAkl
#                    lBxAk1l = lBxAl[c] +               lBx1_AxAkl
#                    
#                    psi0[c][k] = 1/sigma*(lBl[c]-2*lBxAk0l+lAk0l)/lAk0l
#                    psi1[c][k] = 1/sigma*(lBl[c]-2*lBxAk1l+lAk1l)/lAk1l
#        '''           
#        #q_new = np.array(q*step + (1-step)/(1+np.exp(lambd + psi1.sum(0) - psi0.sum(0))), type_t)
#        
#        psi1_psi0 = np.minimum(1/sigma*(psi1 - psi0).sum(0), 30)
#        q_new = np.array(q*step + (1-step)/(1+np.exp(lambd + psi1_psi0)), type_t)
#        
#        '''
#        if np.abs((q_new-q).sum()) < 1e-2:
#            print("Solved at iteration: ", i)
#            return q_new
#        
#        '''
#        diff = np.abs((q_new-q)).mean()
#        if diff  < 1e-6:
#            n_stab += 1
#            if n_stab > 5:
#                print("Solved at iteration: ", i)
#                return q_new
#        else:
#            n_stab = 0        
#        q = q_new
#        
#    print("Solved at max iteration.")     
#    return q 
