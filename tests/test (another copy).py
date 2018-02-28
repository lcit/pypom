#!/usr/bin/env python

import sys
import os
import inspect
import numpy as np

this_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(this_dir)
sys.path.append(parent_dir)
from pypom import utils
from pypom.genpom import Camera, generate_rectangles, POM

here = os.path.dirname(inspect.getfile(inspect.currentframe()))
print(here)

WIDTH = 1110
HEIGHT = 1380
NB_WIDTH = 111
NB_HEIGHT = 138
MAN_RAY = 16
MAN_HEIGHT = 180
ORIGINE_X = 394-1110
ORIGINE_Y = 204-1380

path = "/home/leo/Desktop/20180207/calibration/"
cam0 = Camera.from_json(path+'/gopro6.0/intrinsics.json', path+'/gopro6.0/extrinsics.json', 6)
cam1 = Camera.from_json(path+'/gopro6.1/intrinsics.json', path+'/gopro6.1/extrinsics.json', 6)
cam2 = Camera.from_json(path+'/gopro6.2/intrinsics.json', path+'/gopro6.2/extrinsics.json', 6)
cam3 = Camera.from_json(path+'/gopro6.3/intrinsics.json', path+'/gopro6.3/extrinsics.json', 6)

world_grid = []
for i in range(NB_HEIGHT):
    for j in range(NB_WIDTH):
        step_width = WIDTH/NB_WIDTH
        step_height = HEIGHT/NB_HEIGHT
        world_grid.append([ORIGINE_X + j*step_width,
                           ORIGINE_Y + i*step_height,
                           0])   
world_grid = np.array(world_grid) 

rectangles = generate_rectangles(world_grid, [cam0,cam1,cam2,cam3], MAN_RAY, MAN_HEIGHT, verbose=True)

pom = POM(img_width=320, 
          img_height=180, 
          rectangles=rectangles, 
          input_view_format="/home/leo/Desktop/20180207/videos/gopro6.%c/bg2/bg_%f.png",
          result_view_format="./results/result-f%f-c%c.png",
          result_format="./results/proba-f%f.dat", 
          convergence_view_format="./results/convergence/f%f-c%c-i%i.png", 
          prior=0.0001, 
          sigma_image_density=0.005, 
          max_nb_solver_iterations=100, 
          proba_ignored=1.0, 
          idx_start=0, 
          process=2000,
          p_rect_visible=0.2)

print("N cameras:", pom.n_cams)
print("N positions:", pom.n_poss)

pom.write_to_file(here+'/rectangles.pom')