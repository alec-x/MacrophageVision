# Create dataset
# Alec Xu
# Partial code from Samuel Berryman
# Multi-scale Design Lab
# University of British Columbia
# 2020
from argparse import ArgumentParser as arg_parser

import matplotlib.pyplot as plt
import numpy as np
import os, os.path
import pickle
import sys
import time

def main(raw_args=None):
    parser = arg_parser(description="Threshold each fluorescent channel to derive classes")
    parser.add_argument("path", action="store", type=str, \
                        help="Source path for dir containing raw images.")
    parser.add_argument("label", action="store", type=int, \
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
    num_samples = len(presamples)
    print("Image size: " + str(img_size))
    print("# cells: " + str(num_samples))
    
    arr_shape = (num_samples,) + presamples[0].shape
    arr_data = np.zeros(arr_shape)
    arr_labels = np.zeros(num_samples) + int(args.label)
    # Alveolar Macs: ["mito", "nuclear", "lipid", "bf"]
    # Monocytes: ["mito", "nuclear", "lipid", "bf"]
    # BM Macs: ["lipid", "nuclear", "mito", "bf"] 
    # Alveolar autof: ["green", "red", "blue", "bf"]   
    channel_order = ["green", "red", "blue", "bf"]
    
    print("\nSaving to pickle")
    data = {}
    data["labels"] = arr_labels
    data["channels"] = channel_order
    data["images"] = arr_data
    dir_name = os.path.dirname(args.o)
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    pickle.dump(data, open(args.o, "wb" ))
    
    elapsed = time.time() - start_time
    print("\nTotal time: " + str(elapsed))

    print('\nDataset creation completed')

if __name__=="__main__":
    main()