#!/usr/bin/env python

import sys
import os
import inspect
import numpy as np
from genpom import Camera, generate_rectangles, POM

here = os.path.dirname(inspect.getfile(inspect.currentframe()))
print(here)

WIDTH = 12
HEIGHT = 32
NB_WIDTH = 12
NB_HEIGHT = 30
MAN_RAY = 0.16
MAN_HEIGHT = 1.8
REDUCTION = 1
ORIGINE_X = -3.0
ORIGINE_Y = -6.0

WIDTH = 2.88
HEIGHT = 3.6
NB_WIDTH = 120
NB_HEIGHT = 300
MAN_RAY = 0.16
MAN_HEIGHT = 1.8
REDUCTION = 1
ORIGINE_X = -3.0
ORIGINE_Y = -6.0

cam0 = Camera.from_xml(here+'/intrinsic/intr_CVLab1.xml', here+'/extrinsic/extr_CVLab1.xml', 1.0/100)
cam1 = Camera.from_xml(here+'/intrinsic/intr_CVLab2.xml', here+'/extrinsic/extr_CVLab2.xml', 1.0/100)
cam2 = Camera.from_xml(here+'/intrinsic/intr_CVLab3.xml', here+'/extrinsic/extr_CVLab3.xml', 1.0/100)
cam3 = Camera.from_xml(here+'/intrinsic/intr_CVLab4.xml', here+'/extrinsic/extr_CVLab4.xml', 1.0/100)
cam4 = Camera.from_xml(here+'/intrinsic/intr_IDIAP1.xml', here+'/extrinsic/extr_IDIAP1.xml', 1.0/100)
cam5 = Camera.from_xml(here+'/intrinsic/intr_IDIAP2.xml', here+'/extrinsic/extr_IDIAP2.xml', 1.0/100)
cam6 = Camera.from_xml(here+'/intrinsic/intr_IDIAP3.xml', here+'/extrinsic/extr_IDIAP3.xml', 1.0/100)

world_grid = []
for i in range(NB_HEIGHT):
    for j in range(NB_WIDTH):
        step_width = WIDTH/NB_WIDTH
        step_height = HEIGHT/NB_HEIGHT
        world_grid.append([ORIGINE_X + j*step_width,
                           ORIGINE_Y + i*step_height,
                           0])   
world_grid = np.array(world_grid) 

rectangles = generate_rectangles(world_grid, [cam0,cam1,cam2,cam3,cam4,cam5,cam6], MAN_RAY, MAN_HEIGHT, verbose=True)

pom = POM(img_width=1920, 
          img_height=1080, 
          rectangles=rectangles, 
          input_view_format="./images/cam_%c/img_%f.png",
          result_view_format="./results/result-f%f-c%c.png",
          result_format="./results/proba-f%f.dat", 
          convergence_view_format="./results/convergence/f%f-c%c-i%i.png", 
          prior=0.01, 
          sigma_image_density=0.01, 
          max_nb_solver_iterations=100, 
          proba_ignored=1.0, 
          idx_start=0, 
          process=10,
          p_rect_visible=0.2)

print("N cameras:", pom.n_cams)
print("N positions:", pom.n_poss)

pom.write_to_file(here+'/rectangles.pom')