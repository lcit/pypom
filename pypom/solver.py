#!/usr/bin/env python
""" POM related classes and functions.
"""

import os
from matplotlib.path import Path
os.environ["OMP_NUM_THREADS"] = "1"
import numpy as np
import time
from . import utils
from . import core

__author__ = "Leonardo Citraro" 

X_AXIS = 0
Y_AXIS = 1
Z_AXIS = 2

class Solver(object):
    """Pom solver.

    Parameters
    ----------
    data_loader : DataLoader
        Object in charge of loading images and rectangles
    prior : float
        Prior probability of presence
    sigma : float
        Quality of the background subtraction images
    step : float
        Optimization step. It defines the importance of the preious step w.r.t the new one.
    max_iter : int
        Maximum number of iterations for the optimization.
    tol : float
        Error to reach in order to stop the optimization loop earlier.
    """
    def __init__(self, rectangles, prior=1e-8, sigma=0.001, step=0.9, max_iter=100, tol=1e-6):
        self.rectangles = rectangles
        self.prior = prior
        self.max_iter = max_iter
        self.sigma = sigma
        self.step = step
        self.tol = tol
        
        self.n_cams = len(self.rectangles)
        self.n_positions = len(self.rectangles[0])
        
        e = np.full(self.n_positions, self.prior, np.float32)
        self.exp_lambda = (1-e)/e

        self._rectangles = [self.transform_rectangles_for_c_code(rectangles_view) 
                               for rectangles_view in self.rectangles]
            
    def generate_prior(self):
        return  np.full(self.n_positions, self.prior, np.float32)

    @staticmethod
    def transform_rectangles_for_c_code(rectangles):
        """ Transforms the list of Rectangle objects into the format required by
            the C++ implementation.

        Parameters
        ----------
        rectangles : list of Rectangle objects
            list of Rectangle objects [rect1, rect2, ...]. 
            Make sure the argument Rectangle.visible is set correctly.

        Return
        ------
        _rectangles : list of ints
            list of integers defining the rectangles in the format 
            [ymin1,ymax1,xmin1,xmax1, ymin2,ymax2,xmin2,xmax2, ...].
            In the case the rectangle is not visible the coordinates are set to -1.
        """
        # preparing the rectangles for the cpp wrapper
        _rectangles = []
        for k, r in enumerate(rectangles):
            if r.visible:
                _rectangles += [r.ymin,r.ymax,r.xmin,r.xmax]
            else:
                _rectangles += [-1,-1,-1,-1]
        return np.int32(_rectangles)         
        
    def run(self, idx, B, q=None, verbose=True, debug=False):
        """ Solve for a specific set of images.

        Parameters
        ----------
        idx : int
            Index of the image to load for the views.
        q : 1D numpy array (n_positions,) (optional)
            Probability of presence for each grid location.
        verbose : bool
            Enable status messages 
        debug : bool
            Enable saving convergence images       

        Return
        ------
        qs : list of 1D numpy array [(n_positions,), (n_positions,), ..]
            List of probability of presence for each optimization iteration performed.
        """
        start = time.time()

        if q is None:
            q = self.generate_prior()       
        
        view_shape = B[0].shape[:2]
        n_stab = 0
        qs = []
        diff = -1
        '''
        if debug:
            for c in range(self.n_cams):
                utils.rmdir("./c{}/A".format(c))
                utils.mkdir("./c{}/A".format(c))   
        '''
        for i in range(self.max_iter):  
             
            utils.progress_bar(i, left_msg="[Solver]::Idx {:04d} (diff={:0.2E}<tol) ".format(idx, diff), 
                               right_msg=" Iteration:{}".format(i), 
                               max=self.max_iter-1)
            
            psi_diff = np.zeros((self.n_cams, self.n_positions), np.float32)
                    
            for c in range(self.n_cams): 
                # Equation (31) 
                A_ = core.compute_A_(view_shape, self._rectangles[c], q) # x(1-qkAk) = (1-A)                             
                A = 1-A_ # 1-x(1-qkAk)

                # Integral image (1-A)
                Ai = core.integral_image(A_)

                # Integral image Bx(1-A)
                BAi = core.integral_image(A_*B[c]) # Bx(1-A)
                #BAi[c] = core.integral_image_mask(A_, B[c])
                
                # numpy.sum() is faster than my implementation of the sum in C++
                # this is the reason the solver is implemented in the Python side.
                lAl   = np.sum(A)
                lBxAl = np.sum(B[c]*A)
                lBl   = np.sum(B[c])

                # Equation (34), this outputs the difference psi(B,A1)-psi(B,A0) for one view
                psi_diff[c] = core.compute_psi_diff(Ai, BAi, lAl, lBxAl, lBl, self._rectangles[c], q)

                if debug:
                    points = np.array([(np.mean([r.xmax, r.xmin]), r.ymax) for r in self.rectangles[c] if r.visible])
                    image = np.uint8(255*np.dstack([A, B[c], B[c]/3.0]))
                    image = utils.draw_points(image, points, 1, 'w')
                    utils.save_image("./c{}/A/A_{}_{}.JPG".format(c,idx,i), image)         
            
            # we clamp the difference of distances in order to avoid exp() overflows
            psi1_psi0 = np.minimum(1/self.sigma*psi_diff.sum(0), 30)

            # probability of presence update
            q_new = np.array(q*self.step + (1-self.step)/(1+self.exp_lambda*np.exp(psi1_psi0)), np.float32)
            qs.append(q_new)
            
            # mechanism for early stopping
            diff = np.abs((q_new-q)).mean()
            if diff  < self.tol:
                n_stab += 1
                if n_stab > 5:
                    if verbose:                        
                        utils.progress_bar(i, left_msg="[Solver]::Idx {:04d} (diff={:0.2E}<tol) ".format(idx, diff), 
                                           right_msg=" [{:0.2f}s] Solved at iteration: {} (diff={:0.2E}<tol)".format(time.time()-start, i, diff), 
                                           max=self.max_iter-1, end=True)
                    return qs
            else:
                n_stab = 0 
       
            q = q_new
                
        utils.progress_bar(i, left_msg="[Solver]::Idx {:04d} (diff={:0.2E}<tol) ".format(idx, diff), 
                           right_msg=" [{:0.2f}s] Max iteration reached. (diff={:0.2E}>tol)".format(time.time()-start, i, diff), 
                           max=self.max_iter-1, end=True)
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
