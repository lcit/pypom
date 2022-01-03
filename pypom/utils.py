#!/usr/bin/env python
""" Utility functions.
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
import shutil
import multiprocessing
import itertools
from scipy.linalg import logm, expm

__author__ = "Leonardo Citraro"

class Parallel(object):
    
    def __init__(self, threads=8):
        self.threads = threads
        self.p = multiprocessing.Pool(threads)
        
    def __call__(self, f, iterable, *arg):
        
        if len(arg):
            res = self.p.starmap(f, itertools.product(iterable, *[[x] for x in arg]))
        else:
            res = self.p.map(f, iterable)

        self.p.close()
        self.p.join()
        return res
    
    @staticmethod
    def split_iterable(iterable, n):
        
        if isinstance(iterable, (list,tuple)):
            s = len(iterable)//n
            return [iterable[i:i + s] for i in range(0, len(iterable), s)]
        elif isinstance(iterable, np.ndarray):
            return np.array_split(iterable, n)
        
    @staticmethod
    def join_iterable(iterable):
        
        if isinstance(iterable, (list,tuple)):
            return list(itertools.chain.from_iterable(iterable))
        elif isinstance(iterable, np.ndarray):
            return np.concatenate(iterable)      

def transform_points(H, points):
    # convertPointsFromHomogeneous has some problem when dealing with non float values!
    points = points.astype(np.float)
    
    # if points is just a vector we need to add a new dimension to make it working correctly
    if np.ndim(points) == 1:
        points = points[np.newaxis,:]
    projected_points = cv2.convertPointsFromHomogeneous(np.dot(H, cv2.convertPointsToHomogeneous(points)[:,0,:].T).T)[:,0,:]
    
    return projected_points #(x,y)

def project_KRt(world_points, R, t, K, dist=None):
    rvec = cv2.Rodrigues(R)[0]
    return cv2.projectPoints(world_points, rvec, t, K, None)[0].reshape(-1,2) 

def json_read(filename):
    with open(filename) as f:    
        data = json.load(f)
    return data
        
def json_write(filename, data):
    directory = os.path.dirname(os.path.abspath(filename))
    if not os.path.exists(directory):
        os.makedirs(directory)
    with open(os.path.abspath(filename), 'w') as f:
        json.dump(data, f, indent=2)
        
def yaml_read(filename):
    with open(filename, 'r') as f:
        data = yaml.load(f)
    return data
        
def yaml_write(filename, data):
    directory = os.path.dirname(os.path.abspath(filename))
    if not os.path.exists(directory):
        os.makedirs(directory)
    with open(os.path.abspath(filename), 'w') as f:
        yaml.dump(data, f, default_flow_style=False, width=1000)

def pickle_read(filename):
    with open(filename, "rb") as f:    
        data = pickle.load(f)
    return data
        
def pickle_write(filename, data):
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
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(K, distCoeffs, (w, h), 1.0, (w, h), centerPrincipalPoint=True)    
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

def draw_points(image, centers, radius, color='r'): 
    """ Draws filled point on the image
    """
    _image = image.copy()        
    if color=='r':
        color = [255,0,0]
    elif color=='g':
        color = [0,255,0]
    elif color=='b':
        color = [0,0,255]
    elif color=='w':
        color = [255,255,255]
    elif color=='k':
        color = [0,0,0]
    
    for point in centers:
        _image = cv2.circle(_image, tuple(point.astype(np.int)), radius, color=color, thickness=-1)
    return _image

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

def progress_bar(value, left_msg, right_msg, max, notches=20, end=False):
    if end:
        end="\r\n" 
    else:
        end="\r"
    p=int(value//((max-1)/notches))        
    if p==notches:
        print(left_msg+"|"*(p+2)+right_msg, end=end, flush=True) 
    else:
        print(left_msg+"|"+"|"*(p)+">"+"_"*(notches-p-1)+"|"+right_msg, end=end, flush=True)
        
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