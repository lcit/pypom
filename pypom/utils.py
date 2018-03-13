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
    projected_points = cv2.convertPointsFromHomogeneous(np.dot(H, cv2.convertPointsToHomogeneous(points)[:,0,:].T).T)[:,0,:]
    
    # projected_points is in image pixels.
    # The first column defines the vertical position of the point, the second column 
    # defines the horizonal position as in numpy notation
    return np.hstack([projected_points[:,[1]], projected_points[:,[0]]])

def project_KRt(world_points, R, t, K, dist=None):
    homogeneous = np.dot(K, (np.dot(R, world_points.T) + t.reshape(3,1))).T
    projected_points = homogeneous[:,:2] / homogeneous[:,[2]]  
    if dist is not None:
        projected_points = cv2.undistortPoints(projected_points[:,np.newaxis].astype(np.float32), K, dist) 

    # The first column defines the vertical position of the point in the image, the second column 
    # defines the horizonal position as in numpy notation
    return np.hstack([projected_points[:,[1]], projected_points[:,[0]]])   

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
            elif quality == 'medium':
                imageio.imsave(filename, image, compress_level=4)
            elif quality == 'low':
                imageio.imsave(filename, image, compress_level=9)    
            else:
                imageio.imsave(filename, image)
        elif extension.lower() in [".jpg",".jpeg"]: 
            if quality == 'best':
                imageio.imsave(filename, image, quality=100)
            elif quality == 'medium':
                imageio.imsave(filename, image, quality=75)
            elif quality == 'low':
                imageio.imsave(filename, image, quality=40)
            else:
                imageio.imsave(filename, image)
        elif extension.lower() in [".bmp",".tiff",".gif"]: 
            imageio.imsave(filename, image)                 
        else:
            raise ValueError("Image format ({}) not covered.".format(extension))
            
def downsample_image(img, size, msigma=1.0, interpolation='area'):
    scale_h = size[0]/img.shape[0]
    scale_w = size[1]/img.shape[1]
    
    if interpolation == 'cubic':
        interpolation=cv2.INTER_CUBIC
    elif interpolation == 'area':
        interpolation=cv2.INTER_AREA
    elif interpolation == 'linear':
        interpolation=cv2.INTER_LINEAR
    
    if msigma is not None and msigma > 0:
        img = cv2.GaussianBlur(img, ksize=(0,0), sigmaX=1.0/scale_w*msigma, sigmaY=1.0/scale_h*msigma)
    img = cv2.resize(img, dsize=(size[1], size[0]), interpolation=interpolation)
    return img

def upsample_image(img, size, interpolation='cubic'):

    if interpolation == 'cubic':
        interpolation=cv2.INTER_CUBIC
    elif interpolation == 'area':
        interpolation=cv2.INTER_AREA
    elif interpolation == 'linear':
        interpolation=cv2.INTER_LINEAR
        
    img = cv2.resize(img, dsize=(size[1], size[0]), interpolation=interpolation)
    return img             

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

def progress_bar(value, msg, max, notches=20):   
    p=int(value//((max-1)/notches))    
    print("|"+"|"*(p)+">"+"_"*(notches-p)+"|"+msg, end="\r", flush=True)
    if p==notches:
        print("|"*(p+2), end="\r", flush=True) 
        
def scale_homography_left(H, scale):
    S = np.array([[scale, 0, 0], [0, scale, 0], [0,0,1]])
    return np.dot(S, H)

def scale_homography_right(H, scale):
    S = np.array([[scale, 0, 0], [0, scale, 0], [0,0,1]])
    return np.dot(H, S)

def rotmatrix_to_rotvector(R):
    return logm(R)[[2, 0, 1], [1, 2, 0]]

def rotvector_to_rotmatrix(rotvector):
    a, b, c = rotvector
    skew = np.array([[0, -c, b], [c, 0, -a], [-b, a, 0]])
    return expm(skew)

def invert_Rt(R, t):
    Ri = R.T
    ti = np.dot(-Ri, t)
    return Ri, ti

def homography_from_Rt(R, t, K):
    """ Retrieves homography matrix from rotaton and translation.
    Also, make sure R and t are in the right verso, otherwise use invert_Rt()
    """
    return np.dot(K, np.hstack([R[:,0:2], t]))

unit_conversion  =  {"cm":{"cm":lambda x: x,
                           "m":lambda x: x*0.01,
                           "ft":lambda x: x*0.0328084,
                           "in":lambda x: x*0.393701},
                     "m" :{"cm":lambda x: x*100,
                           "m":lambda x: x,
                           "ft":lambda x: x*3.28084,
                           "in":lambda x: x*39.3701},
                     "ft":{"cm":lambda x: x*30.48,
                           "m":lambda x: x*0.3048,
                           "ft":lambda x: x,
                           "in":lambda x: x*12},
                     "in":{"cm":lambda x: x*2.54,
                           "m":lambda x: x*0.0254,
                           "ft":lambda x: x*0.0833333,
                           "in":lambda x: x}}

def value_unit_conversion(value, in_unit="cm", out_unit="cm"):
    return unit_conversion[in_unit][out_unit](value)

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








