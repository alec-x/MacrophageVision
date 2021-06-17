# Create dataset
# Alec Xu
# Partial code from Samuel Berryman
# Multi-scale Design Lab
# University of British Columbia
# 2020
from argparse import ArgumentParser as arg_parser

from binarize_otsu import binarize_otsu
import cv2
import csv
from collections import defaultdict
import math
import matplotlib.pyplot as plt
import numpy as np
import os, os.path
import pandas as pd
import pickle
from PIL import Image
import sys
import time

def main(raw_args=None):
    parser = arg_parser(description="Threshold each fluorescent channel to derive classes")
    parser.add_argument("path", action="store", type=str, \
                        help="Source path for dir containing raw images.")
    parser.add_argument("label", action="store", type=str, \
                        help="label for class")
    parser.add_argument("-o", action="store", type=str, \
                        help="out path of dataset (default=current dir")
    # Default optional args
    parser.set_defaults(o=".\\default_thresholded", p=False)
    
    args = parser.parse_args(raw_args)
    start_time = time.time()
    print("Thresholding dataset with params...") 
    print("Input path: " + args.path)
    print("Output path: " + args.o)
    
    if os.path.exists(args.o): 
        sys.exit("output path exists... exiting")
    
    # Load Data
    print("\nLoading pickle")
    presamples = None
    with open(args.path, 'rb') as handle:
        presamples = pickle.load(handle)
    img_size = presamples[0][0].shape[0]
    print("Image size: " + str(img_size))
    print("# candidate cells: " + str(len(presamples)))
    print("\nMaking directories")
    
    # Alveolar Macs: ["mito", "nuclear", "lipid", "bf"]
    # Monocytes: ["mito", "nuclear", "lipid", "bf"]
    # BM Macs: ["lipid", "nuclear", "mito", "bf"] 
    # Alveolar autof: ["green", "red", "blue", "bf"]   
    img_type = ["green", "red", "blue", "bf"]
    os.makedirs(args.o)
    for channel in img_type:
        os.makedirs(args.o + "\\" + channel)
    os.makedirs(args.o + "\\mito")
    os.makedirs(args.o + "\\lipid")
    os.makedirs(args.o + "\\nuclear")

    
    print("\nThresholding samples")
    output = []  
    num_samples = len(presamples)
    total_sig = math.ceil(math.log10(num_samples))
    load_inc = int(num_samples / 100)
    i = 0
    for sample in presamples:
        curr_output = []
        for j in range(sample.shape[0]):
            tmp_img = Image.fromarray(sample[j])
            channel = img_type[j]
            curr_num = str(i).rjust(total_sig,'0')
            path = channel + "\\" + curr_num + channel + ".png"
            tmp_img.save(args.o + "\\" + path)
            curr_output.append(path)
        try:
            pass
        except Exception as e:
            print(e)
            continue
        if i % load_inc == 0:
            print(f'Sample {i}/{num_samples}', end="\r")
        i += 1
        curr_output.append(args.label)
        output.append(curr_output)

    print("\nSaving paths")
    col_names = ["mito", "nuclear", "lipid", "bf", "label"]
    output = pd.DataFrame.from_records(output, columns=col_names)
    output.to_csv(args.o + "labels.csv", index=False)

    elapsed = time.time() - start_time
    print("\nTotal time: " + str(elapsed))

    print('\nDataset creation completed')

if __name__=="__main__":
    main()