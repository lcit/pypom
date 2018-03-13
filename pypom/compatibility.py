import numpy as np
import sys
import os
from . import pom
from . import utils

'''    
def to_string(self):     
    if self.visible:
        return "RECTANGLE {self.cam} {self.idx} {self.xmin} {self.ymin} {self.xmax} {self.ymax}".format(self=self)
    else:
        return "RECTANGLE {self.cam} {self.idx} notvisible".format(self=self)
'''

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
        lines = f.read_lines()
    
    cam_ = []
    #idx_ = []
    rectangles_ = []
    for line in lines:  
        if "RECTANGLE" in line:
            cam, idx, visible, xmin, ymin, xmax, ymax = parse_rectangle_line(line)            
            cam_.append(cam)
            #idx_.append(idx) # I assume the indexes are sorted already
            rectangles_.append(pom.Rectangle(xmin=xmin, ymin=ymin, xmax=xmax, ymax=ymax, visible=visible)) 
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
    camsid = utils.sort_nicely(set(cam_))
    rectangles = [rectangles_[np.where(cam_==c)[0]] for c in camsid]
    
    return rectangles, view_shape, n_cams, n_positions, prior, sigma, max_iter
    
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
        utils.mkdir(os.path.dirname(filename))
        
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