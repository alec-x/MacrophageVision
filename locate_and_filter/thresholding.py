# Create dataset
# Alec Xu
# Partial code from Samuel Berryman
# Multi-scale Design Lab
# University of British Columbia
# 2020
from argparse import ArgumentParser as arg_parser
from binarize_otsu import binarize_otsu
import cv2
from collections import defaultdict
import math
import matplotlib.pyplot as plt
import numpy as np
import os, os.path
import pickle
from PIL import Image
import sys
import time

# Do definitions of validity in function for now. Move to config file later
def threshold(sample, box_size, min_size=30):
    try:
        img_bf = np.copy(sample[0])
        img_mito = np.copy(sample[1])
        
        img_cd80 = np.copy(sample[2])
        img_cd206 = np.copy(sample[3])

        mask_mito = binarize_otsu(img_mito, min_size).astype(np.uint8)
        kernel = np.ones((7, 7), np.uint8)
        mask_erode = cv2.dilate(mask_mito, kernel, iterations=1)

        cd80_relevant = np.multiply(mask_mito,img_cd80).flatten()
        cd80_relevant = list(filter(lambda num: num != 0, cd80_relevant))
        cd80_avg = np.percentile(cd80_relevant,80)
        
        cd206_relevant = np.multiply(mask_mito,img_cd206).flatten()
        cd206_relevant = list(filter(lambda num: num != 0, cd206_relevant))
        cd206_avg = np.percentile(cd206_relevant,80)  
        
        cd80_background = np.median(np.multiply(1-mask_mito,img_cd80))
        cd206_background = np.median(np.multiply(1-mask_mito,img_cd206))    

        cd80_thresh = cd80_avg - cd80_background
        cd206_thresh = cd206_avg - cd206_background
        return cd80_thresh, cd206_thresh

    except Exception as e:
        print(e)
        raise e

    return True 
    
def main(raw_args=None):
    parser = arg_parser(description="Create dataset from cell images in dir. \
        \n Input requires a directory with images under directories with \
        corresponding cell name")
    parser.add_argument("path", action="store", type=str, \
                        help="Source path for dir containing raw images.")
    parser.add_argument("-o", action="store", type=str, \
                        help="out path of dataset (default=current dir")
    parser.add_argument("-i", dest="n", action="store_true")

    # Default optional args
    parser.set_defaults(o=".\\data", p=False, i=False)
    
    args = parser.parse_args(raw_args)
    start_time = time.time()
    print("Thresholding dataset with params...") 
    print("Input path: " + args.path)
    print("Output path: " + args.o)
    
    if os.path.exists(args.o): 
        sys.exit("output path exists... exiting")

    img_type = ["BF", "mito", "cd80", "cd206"]
    
    # Load Data
    print("\nLoading pickle")
    presamples = None
    with open(args.path, 'rb') as handle:
        presamples = pickle.load(handle)
    img_size = presamples[0][0].shape[0]

    print("Image size: " + str(img_size))
    print("# candidate cells: " + str(len(presamples)))
    print("\nMaking directories")
    
    os.makedirs(args.o)
    for channel in img_type:
        os.mkdir(args.o + "\\" + channel)

    print("\nThresholding samples")
    output = []  
    num_samples = len(presamples)
    total_sig = math.ceil(math.log10(num_samples))
    load_inc = int(num_samples / 100)
    i = 0
    for sample in presamples:
        
        cd80_t, cd206_t = threshold(sample, img_size)
        curr_output = [i, cd80_t, cd206_t]
        for j in range(sample.shape[0]):
            tmp_img = Image.fromarray(sample[j])
            channel = img_type[j]
            curr_num = str(i).rjust(total_sig,'0')
            path = args.o + "\\" + channel + "\\" + curr_num + channel + ".png"
            tmp_img.save(path)
            curr_output.append(path)
        try:
            pass
        except Exception as e:
            print(e)
            continue
        if i % load_inc == 0:
            print(f'Sample {i}/{num_samples}', end="\r")
        i += 1
        
        output.append(curr_output)

    print("\nSaving thresholds")
    output = np.stack(output, axis=0)
    pickle.dump(output, open(args.o + "\\data.pickle", "wb" ))

    elapsed = time.time() - start_time
    print("\nTotal time: " + str(elapsed))

    print('\nDataset creation completed')

if __name__=="__main__":
    main()