#!/usr/bin/env python
""" Utility function.
"""

import numpy as np
import cv2
import os
import imageio
import glob
import re
import json
import yaml
import pickle
import torch
import shutil
from torch.autograd import Variable
from scipy.linalg import logm, expm

__author__ = "Leonardo Citraro"
__email__ = "leonardo.citraro@epfl.ch"

def transform_points(H, points):
    # convertPointsFromHomogeneous has some problem when dealing with non float values!
    points = points.astype(np.float)
    
    # if points is just a vector we need to add a new dimension to make it working correctly
    if np.ndim(points) == 1:
        points = points[np.newaxis,:]
    return cv2.convertPointsFromHomogeneous(np.dot(H, cv2.convertPointsToHomogeneous(points)[:,0,:].T).T)[:,0,:]

def json_read(filename):
    with open(filename) as f:    
        data = json.load(f)
    return data
        
def json_write(data, filename):
    directory = os.path.dirname(os.path.abspath(filename))
    if not os.path.exists(directory):
        os.makedirs(directory)
    with open(os.path.abspath(filename), 'w') as f:
        json.dump(data, f, indent=2)
        
def yaml_read(filename):
    with open(filename, 'r') as f:
        data = yaml.load(f)
    return data
        
def yaml_write(data, filename):
    directory = os.path.dirname(os.path.abspath(filename))
    if not os.path.exists(directory):
        os.makedirs(directory)
    with open(os.path.abspath(filename), 'w') as f:
        yaml.dump(data, f, default_flow_style=False, width=1000)

def pickle_read(filename):
    with open(filename, "rb") as f:    
        data = pickle.load(f)
    return data
        
def pickle_write(data, filename):
    directory = os.path.dirname(os.path.abspath(filename))
    if not os.path.exists(directory):
        os.makedirs(directory)
    with open(filename, 'wb') as f:
        pickle.dump(data, f)         

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
        
def rmdir(directory):
    directory = os.path.abspath(directory)
    if os.path.exists(directory): 
        shutil.rmtree(directory)        
        
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

def rgb2gray(image):
    dtype = image.dtype
    gray = np.dot(image[...,:3], [0.299, 0.587, 0.114])
    return gray.astype(dtype)

def load_image(filename, gray=False):
    if os.path.isfile(filename):
        image = imageio.imread(filename)
        
        # in the case the image is a PNG with 4 channels
        # we dump the last channel which is the "transparent"
        if np.ndim(image)==3:
            if gray:
                return rgb2gray(image[:,:,:3])
            else:
                return image[:,:,:3]
        else:
            return image
    else:
        raise ValueError("Filename doesn't exists! - {}".format(filename))
    
def save_image(filename, image, quality='best'):
    if image is not None:
        directory = os.path.dirname(os.path.abspath(filename))
        if not os.path.exists(directory):
            os.makedirs(directory)

        _, extension = os.path.splitext(filename)
        if extension.lower() == '.png': 
            if quality == 'best':
                imageio.imsave(filename, image, compress_level=0)
            elif quality == 'low':
                imageio.imsave(filename, image, compress_level=9)    
            else:
                imageio.imsave(filename, image)
        elif extension.lower() in [".jpg",".jpeg",".bmp",".tiff",".png",".gif"]: 
            imageio.imsave(filename, image)                 
        else:
            raise ValueError("Image format ({}) not covered.".format(extension))

def compute_edges(image, th0=None, th1=None, equalization=True):
    if np.ndim(image)==2:
        gray = image
    else:
        gray = rgb2gray(image)
    
    if equalization:
        gray = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8)).apply(gray)
       
    if th0 is None or th1 is None:
        ret, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        edges = cv2.Canny(gray, ret*0.5, ret, apertureSize=3, L2gradient=True)
    else:
        edges = cv2.Canny(gray, th0, th1, apertureSize=3, L2gradient=True)
    return edges 

def gaussian_blurring(image, sigma):
    # GaussianBlur's speed depends a lot on the type of the array:
    # image = image.astype(np.uint8)# baseline
    # image = image.astype(np.float)# float64 ~8x slower
    # image = image.astype(np.float32)# float32 ~1.7x slower
    image = image.astype(np.float32)
    if sigma == 0:
        return image    
    blurred = cv2.GaussianBlur(image, (0, 0), sigma)
    return blurred

def dict_try_retrieve(json_data, key):
    try:
        return json_data[key]
    except:
        return None

# not really the best search algorithm invented but gives some flexibility
def search_in_dict(dict, exact=None, groups=None):    
    if exact is not None:
        for key, value in dict.items():
            if key==exact:
                return value

    if groups is not None:
        for group in groups:
            re_expression = ''.join(["{}|".format(x) for x in group])
            re_expression = re_expression[:-1] # we remove the last or "|"
            for key, value in dict.items():
                if len(set(re.findall(re_expression, key, flags=re.I)))==len(group):
                    return value
    return None

def retrieve_intrinsics_from_json(filename):
    calibration = json_read(filename)

    K = search_in_dict(calibration, "K", [("mtx"),("proj")])
    if K is None:
        raise RuntimeError("Unable to retrieve 'K' from file {}".format(filename))
        
    dist = search_in_dict(calibration, "distCoeffs", [("dist")])      
    if dist is None:
        raise RuntimeError("Unable to retrieve 'distCoeffs' from file {}".format(filename)) 
        
    image_shape = search_in_dict(calibration, "image_shape", [("im","shape"), ("im","size"), ("size"), ("shape")])
    if image_shape is None:
        raise RuntimeError("Unable to retrieve 'image_shape' from file {}".format(filename))        

    return np.array(K).reshape(3,3), np.array(dist).ravel(), np.array(image_shape).ravel()

def retrieve_image_and_world_points_from_json(filename):
    points = json_read(filename)
    
    image_points = search_in_dict(points, "image_points", [("image","point")])
    if image_points is None:
        raise RuntimeError("Unable to retrieve 'image_points' from file {}".format(filename))
        
    world_points = search_in_dict(points, "world_points", [("world","point"), ("model","point")])
    if world_points is None:
        raise RuntimeError("Unable to retrieve 'world_points' from file {}".format(filename))
        
    image_shape = search_in_dict(points, "image_shape", [("im","shape"), ("im","size"), ("size"), ("shape")])
    if image_shape is None:
        raise RuntimeError("Unable to retrieve 'image_shape' from file {}".format(filename))
        
    unit = search_in_dict(points, "unit", [("unit")])
    if unit is None:
        raise RuntimeError("Unable to retrieve 'unit' from file {}".format(filename))
        
    undistorted = search_in_dict(points, "distCoeffs", [("undist")])
    if undistorted is None:
        raise RuntimeError("Unable to retrieve 'undistorted' from file {}".format(filename))
        
    return np.array(image_points), np.array(world_points), np.array(image_shape).ravel(), unit, undistorted

def retrieve_extrinsics_from_json(filename):
    calibration = json_read(filename)
    
    R = search_in_dict(calibration, "R", [("Rot")])
    if R is None:
        raise RuntimeError("Unable to retrieve 'R' from file {}".format(filename))
        
    t = search_in_dict(calibration, "t", [("trans")])
    if t is None:
        raise RuntimeError("Unable to retrieve 't' from file {}".format(filename))

    image_shape = search_in_dict(calibration, "image_shape", [("im","shape"), ("im","size"), ("size"), ("shape")])
    if image_shape is None:
        raise RuntimeError("Unable to retrieve 'image_shape' from file {}".format(filename))
        
    unit = search_in_dict(calibration, "unit", [("unit")])
    if unit is None:
        raise RuntimeError("Unable to retrieve 'unit' from file {}".format(filename))                

    return np.array(R).reshape(3,3), np.array(t).reshape(3,1), np.array(image_shape).ravel(), unit
        
def scale_homography(H, scale):
    S = np.array([[scale, 0, 0], [0, scale, 0], [0,0,1]])
    return np.dot(S, H)

def rotmatrix_to_rotvector(R):
    return logm(R)[[2, 0, 1], [1, 2, 0]]

def rotvector_to_rotmatrix(rotvector):
    a, b, c = rotvector
    skew = np.array([[0, -c, b], [c, 0, -a], [-b, a, 0]])
    return expm(skew)

# https://c4science.ch/source/posenet/browse/master/unet_utils.py
# Pablo Marquez Neila 
def _decompose_homography(homography, K=np.eye(3)):
    M = np.dot(np.linalg.inv(K), homography)

    M /= M[2, 2]

    scale = (np.linalg.norm(M[:, 0]) + np.linalg.norm(M[:, 1])) / 2
    M2 = M / scale
    r3 = np.cross(M2[:, 0], M2[:, 1])
    R = np.array([M2[:, 0], M2[:, 1], r3]).T
    t = M2[:, 2]

    # Orthogonalize R
    U, D, V = np.linalg.svd(R)
    R = np.dot(U, V)

    return R, t

# https://c4science.ch/source/posenet/browse/master/unet_utils.py
# Pablo Marquez Neila 
def _rodrigues(w):
    """A PyTorch implementation of the Rodrigues formula."""

    theta = torch.sqrt(torch.sum(w ** 2))
    sin_theta = torch.sin(theta)
    cos_theta = torch.cos(theta)

    a, b, c = w[0] / theta, w[1] / theta, w[2] / theta

    zero = Variable(torch.Tensor([0]))
    zero.requires_grad = False

    K = torch.stack([torch.cat([zero, -c, b]),
                     torch.cat([c, zero, -a]),
                     torch.cat([-b, a, zero])], dim=0)

    eye = Variable(torch.eye(3))
    eye.requires_grad = True

    return eye + sin_theta * K + (1 - cos_theta) * torch.matmul(K, K)

# https://c4science.ch/source/posenet/browse/master/unet_utils.py
# Pablo Marquez Neila 
def _transform_points(projection, rotation, translation, points):
    R = _rodrigues(rotation)
    transformed_points = torch.matmul(points, R.t()) + translation
    proj = torch.matmul(transformed_points, projection.t())
    return torch.stack([proj[:, 0] / proj[:, 2], proj[:, 1] / proj[:, 2]], dim=1)

# https://c4science.ch/source/posenet/browse/master/unet_utils.py
# Pablo Marquez Neila 
def _minimize_reprojection_error_LM(K, keypoints, gt_points,
                                   init_w, init_t, eta=0.0, tol=1e-3, num_iters=50):

    def to_torch(x):
        return Variable(torch.from_numpy(np.float32(x)))

    def from_torch(x):
        return x.data.cpu().numpy()

    projection = to_torch(K)
    points = to_torch(keypoints)
    gt_points = to_torch(gt_points)
    w = to_torch(init_w)
    t = to_torch(init_t)
    w.requires_grad = True
    t.requires_grad = True

    for i in range(num_iters):

        proj_points = _transform_points(projection, w, t, points)
        residuals = (proj_points - gt_points).view(-1)

        J = []
        for i in range(residuals.size(0)):
            aux = np.zeros(residuals.size(0))
            aux[i] = 1
            J_i = torch.autograd.grad([residuals], [w, t], to_torch(aux), retain_graph=True)
            J.append(torch.cat(J_i))

        J = torch.stack(J)
        Jr = torch.matmul(J.t(), residuals)
        JJ_eta = torch.matmul(J.t(), J) + eta * Variable(torch.eye(6))

        # TODO: Is it possible to solve this in PyTorch?
        step_wt = np.linalg.solve(from_torch(JJ_eta), -from_torch(Jr))

        w.data.add_(torch.from_numpy(step_wt[:3]))
        t.data.add_(torch.from_numpy(step_wt[3:]))

        if np.linalg.norm(step_wt) < tol:
            break

    return from_torch(w), from_torch(t)

# https://c4science.ch/source/posenet/browse/master/unet_utils.py
# Pablo Marquez Neila 
def _visible_points_for_homography(homography, image_shape, points):
    """
    Determine the subset of points from `points` that are visible in an image after
    a homography transformation.

    Returns the set of visible points both in the original frame of reference
    and after the homography transformation.
    """
    proj_points = transform_points(homography, points[:, :2])
    mask = np.ones(len(proj_points))
    mask = np.logical_and(mask, proj_points[:, 0] >= 0)
    mask = np.logical_and(mask, proj_points[:, 1] >= 0)
    mask = np.logical_and(mask, proj_points[:, 0] < image_shape[1])
    mask = np.logical_and(mask, proj_points[:, 1] < image_shape[0])

    return points[mask], proj_points[mask]

# https://c4science.ch/source/posenet/browse/master/unet_utils.py
# Pablo Marquez Neila 
def homography_to_transformation(homography, K, model_points, image_shape):

    # Get an initial estimation of R and t.
    R, t = _decompose_homography(homography, K)

    # Compute the rotation vector
    w = rotmatrix_to_rotvector(R)

    # For the minimization of the reprojection error, we use only the points
    # of the model that are visible in the image.
    visible_model_points, visible_projected_points = _visible_points_for_homography(homography,
                                                                                    image_shape,
                                                                                    model_points)

    # Fine-tune rotation and translation to minimize the reprojection error
    neww, newt = _minimize_reprojection_error_LM(K,
                                                 visible_model_points,
                                                 visible_projected_points,
                                                 w, t,
                                                 eta=0)

    # Reconstruct the rotation matrix from the rotation vector.
    newR = rotvector_to_rotmatrix(neww)

    return newR, newt








