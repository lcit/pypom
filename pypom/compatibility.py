import numpy as np
import sys
import os
from . import pom
from . import utils

__author__ = "Leonardo Citraro" 
    
def compose_rectangle_line(cam, idx, visible, xmin=None, xmax=None, ymin=None, ymax=None):  
    
    if visible:
        return "RECTANGLE {cam} {idx} {xmin} {ymin} {xmax} {ymax}".format(cam=cam, idx=idx, 
                                                                          ymin=ymin, ymax=ymax, 
                                                                          xmin=xmin, xmax=xmax)
    else:
        return "RECTANGLE {cam} {idx} notvisible".format(cam=cam, idx=idx)

def parse_rectangle_line(string):
    if "RECTANGLE" not in string:
        raise ValueError("The string has to start with RECTANGLE")
    elements = string.strip().split(' ')
    cam = int(elements[1])
    idx = int(elements[2])
    if "not" in elements[3]:
        visible = False
        xmin = None
        ymin = None
        xmax = None
        ymax = None 
    else:
        visible = True
        xmin = int(elements[3])
        ymin = int(elements[4])
        xmax = int(elements[5])
        ymax = int(elements[6]) 
    return cam, idx, visible, xmin, ymin, xmax, ymax

def parse_room_line(string):
    if "ROOM" not in string:
        raise ValueError("The string has to start with ROOM")
    elements = string.strip().split(' ')
    view_width  = int(elements[1])
    view_height = int(elements[2])
    n_cams      = int(elements[3])
    n_positions = int(elements[4])
    return view_width, view_height, n_cams, n_positions

def read_pom_file(filename):
    
    with open(filename, "r") as f:
        lines = list(f)
        
    n_cams = None
    n_positions = None
    view_shape = None
    prior = None
    sigma = None
    max_iter = None
    
    cam_ = []
    #idx_ = []
    rectangles_ = []
    for line in lines:  
        if "RECTANGLE" in line:
            cam, idx, visible, xmin, ymin, xmax, ymax = parse_rectangle_line(line)    
            cam_.append(cam)
            #idx_.append(idx) # I assume the indexes are sorted already
            rectangle = pom.Rectangle(xmin=xmin, ymin=ymin, xmax=xmax, ymax=ymax, 
                                      visible=visible, ID=idx)
            rectangles_.append(rectangle) 
        elif "ROOM" in line: 
            view_width, view_height, n_cams, n_positions = parse_room_line(line)
            view_shape = (view_height, view_width)
        elif "INPUT_VIEW_FORMAT" in line:
            elements = line.strip().split(' ')
            input_view_format = elements[1]
        elif "RESULT_VIEW_FORMAT" in line:    
            elements = line.strip().split(' ')
            result_view_format = elements[1]            
        elif "RESULT_FORMAT" in line:             
            elements = line.strip().split(' ')
            result_format = elements[1]             
        elif "PRIOR" in line:             
            elements = line.strip().split(' ')
            prior = float(elements[1])
        elif "SIGMA_IMAGE_DENSITY" in line: 
            elements = line.strip().split(' ')
            sigma = float(elements[1])            
        elif "MAX_NB_SOLVER_ITERATIONS" in line: 
            elements = line.strip().split(' ')
            max_iter = int(elements[1])                          
        elif "PROBA_IGNORED" in line: 
            elements = line.strip().split(' ')
            proba_ignored = float(elements[1])                         
        elif "PROCESS" in line: 
            elements = line.strip().split(' ')
            idx_start   = int(elements[1])
            run_for     = int(elements[2])  

    cam_ = np.array(cam_)
    rectangles_ = np.array(rectangles_)   
    
    cam_indexes = np.sort(np.unique(cam_))

    rectangles = [rectangles_[np.where(cam_==c)[0]].tolist() for c in cam_indexes]
    
    if n_positions is None:
        n_positions = len(np.where(cam_==cam_[0])[0])
    if n_cams is None:
        n_cams = len(cam_indexes)
    
    return rectangles, view_shape, n_cams, n_positions, prior, sigma, max_iter

def write_pom_file(filename, img_width, img_height, n_cams, n_positions, rectangles,
                    input_view_format="./images/cam_%c/img_%f.png", 
                    result_view_format="./results/result-f%f-c%c.png",
                    result_format="./results/proba-f%f.dat", 
                    convergence_view_format="./results/convergence/f%f-c%c-i%i.png", 
                    prior=0.01, sigma_image_density=0.01, max_nb_solver_iterations=100, 
                    proba_ignored=1.0, idx_start=0, process=10):
    
    utils.mkdir(os.path.dirname(filename))

    text_file = open(filename, "w")
    text_file.write("ROOM {} {} {} {}\n\n".format(img_width, img_height, n_cams, n_positions))
    for cam in range(n_cams):
        for idx in range(n_positions):
            rect = rectangles[cam][idx]                
            line = compose_rectangle_line(cam, idx, rect.visible, rect.xmin, rect.xmax, rect.ymin, rect.ymax)                
            text_file.write(line+"\n")       
    text_file.write("\n")

    text_file.write("INPUT_VIEW_FORMAT {}\n\n".format(input_view_format))
    if result_view_format is not None:
        text_file.write("RESULT_VIEW_FORMAT {}\n\n".format(result_view_format))  
    if result_format is not None:
        text_file.write("RESULT_FORMAT {}\n\n".format(result_format))             
    if convergence_view_format is not None:
        text_file.write("CONVERGENCE_VIEW_FORMAT {}\n\n".format(convergence_view_format))    
    if prior is not None:
        text_file.write("PRIOR {}\n".format(prior))   
    if sigma_image_density is not None:
        text_file.write("SIGMA_IMAGE_DENSITY {}\n\n".format(sigma_image_density)) 
    if max_nb_solver_iterations is not None:
        text_file.write("MAX_NB_SOLVER_ITERATIONS {}\n\n".format(max_nb_solver_iterations)) 
    if proba_ignored is not None:
        text_file.write("PROBA_IGNORED {}\n\n".format(proba_ignored)) 
    if idx_start is not None and process is not None:
        text_file.write("PROCESS {} {}\n\n".format(idx_start, process))             

    text_file.close() 
    
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
        utils.mkdir(os.path.dirname(self.input_view_format))        
        
        self.result_view_format = result_view_format
        self._check_png(self.result_view_format)
        utils.mkdir(os.path.dirname(self.result_view_format)) 
           
        self.result_format = result_format
        utils.mkdir(os.path.dirname(self.result_format))
            
        self.convergence_view_format = convergence_view_format  
        self._check_png(self.convergence_view_format)
        utils.mkdir(os.path.dirname(self.convergence_view_format))
        
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
        write_pom_file(filename, self.img_width, self.img_height, self.n_cams, self.n_poss, self.rectangles,
                       self.input_view_format, self.result_view_format,
                       self.result_format, self.convergence_view_format, 
                       self.prior, self.sigma_image_density, self.max_nb_solver_iterations, 
                       self.proba_ignored, self.idx_start, self.process)        