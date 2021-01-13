#!/usr/bin/env python3
import cv2
import argparse
import os
import sys
import inspect

from pypom import utils

__author__ = "Leonardo Citraro"

def main(filename="example.avi",
         start=0,
         stop=1,
         step=1,
         out=None,
         downsampling=1):
    """ Extraction of a sequence of frame from a video file.

    This script require Opencv with FFmpeg support.
    
    Parameters
    ----------
    filename: str
        filename of the video file
    start: int
        begin of the sequence
    stop: int
        end of the sequence
    step: int
        how many frames to skip at each iteration
    out: str
        folder where to save the frames
    downsampling: int
        downsampling factor

    Example
    ----------
    python3 extract_sequence_of_frames.py --filaname video.mp4 --start 0 --stop 2000 --step 2
    """
    
    filename = os.path.abspath(filename)
    basename = os.path.basename(filename).split('.')[0]
    
    if out is None:
        out = os.getcwd() + "/{}_start{start}_stop{stop}_step{step}/".format(basename, **locals())

    print("Input file: {}".format(filename))
    print("Ouput folder: {}".format(out))
    
    utils.mkdir(os.path.dirname(out))
    
    cap = cv2.VideoCapture()    
    ret = cap.open(filename)
    if ret:
        
        cap.set(cv2.CAP_PROP_POS_FRAMES, start)
        while(cap.isOpened()):

            ret, frame = cap.read()
            frame_number = int(cap.get(cv2.CAP_PROP_POS_FRAMES))

            if not ret:
                print("!!Unable to read frame {}".format(frame_number))
                continue
            
            if frame_number >= stop:
                print("Exit at frame {}".format(frame_number))
                break
            
            if frame_number >= start and frame_number%step==0: 
                if downsampling > 1:
                    frame = cv2.resize(frame, None, None, 1.0/downsampling, 1.0/downsampling, cv2.INTER_AREA)
                cv2.imwrite(out + "frame_{}_{}_{}.JPG".format(start, step, frame_number), frame)
                print("Saving frame {}".format(frame_number))
        
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    else:
        print("Unable to open the file.")
        
    cap.release()

    print("Done.")

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--filename", "-f", type=str, required=True)
    parser.add_argument("--start", "-b", type=int, default=0, required=False)
    parser.add_argument("--stop", "-e", type=int, default=-1, required=False)
    parser.add_argument("--step", "-s", type=int, default=1, required=False)
    parser.add_argument("--out", "-o", type=str, default=None, required=False)
    parser.add_argument("--downsampling", "-d", type=int, default=1, required=False)

    args = parser.parse_args()
    main(**vars(args))    
