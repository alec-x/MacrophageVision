# Create dataset
# Alec Xu
# Partial code from Samuel Berryman
# Multi-scale Design Lab
# University of British Columbia
# 2020
from argparse import ArgumentParser as arg_parser
from binarize_otsu import binarize_otsu
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
import os, os.path
import pickle
from PIL import Image
from scipy import ndimage as ndi
from skimage import filters
import sys
import time
import cv2

# Do definitions of validity in function for now. Move to config file later
def sample_valid(sample, box_size, failures, invalids):
    
    img_mito = np.copy(sample[1])
    """
    img_bf = np.copy(sample[0])
    img_cd80 = np.copy(sample[2])
    img_cd206 = np.copy(sample[3])
    """
    min_size = 30
    
    mask_mito = binarize_otsu(img_mito, min_size)
    kernel = np.ones((3, 3), np.uint8)
    mask_erode = cv2.erode(mask_mito.astype(np.uint8), kernel, iterations=2)

    # Allow diagonal connections to count as one object
    connectivity = np.array(np.ones((3,3)), dtype=np.bool)
    _, num_objs = ndi.label(mask_erode, connectivity)

    # Check if mitochondria stain is too dim
    if np.percentile(img_mito, 99) - np.percentile(img_mito, 5) < 20:
        failures["mito_dim"] += 1
        invalids["mito_dim"].append(sample)
        return False
            
    # Number of cells in sample using mito stain
    # After this, mask_mito should only have 0 = background 1 = mito location
    if num_objs > 1:
        failures["num_obj"] += 1
        invalids["num_obj"].append(sample)
        return False        

    # Size of mito stain in range cell size
    mito_size = np.sum(mask_mito) # num px
    if mito_size < 150:
        failures["mito_size_small"] += 1
        invalids["mito_size_small"].append(sample)
        return False

    if mito_size > 2000:
        failures["mito_size_large"] += 1
        invalids["mito_size_large"].append(sample)
        return False

    return True 
    
def main(raw_args=None):
    parser = arg_parser(description="Basic filtering to remove invalid cell \
                        samples")
    parser.add_argument("path", action="store", type=str, \
                        help="Source path for dir containing raw images.")
    parser.add_argument("-o", action="store", type=str, \
                        help="out path of dataset")
    parser.add_argument("-i", dest="i", action="store_true", \
                        help="output invalid samples instead of valid samples")

    # Default optional args
    parser.set_defaults(s=96, o=".\\default_filtered", p=False, i=False, b=False)
    
    args = parser.parse_args(raw_args)
    
    print("filtering dataset with params...") 
    print("Image size: " + str(args.s) + "px")
    print("Input path: " + args.path)
    print("Output path: " + args.o)
    print("Output invalid samples instead: " + str(args.i))

    if os.path.exists(args.o): 
        sys.exit("output path exists... exiting")

    print("\nLoading pickle")
    start_time = time.time()
    
    # Create dict of lists to hold each label type
    #label_types = set(labels)
    unfiltered_samples = None
    filtered_samples = []
    invalid_samples = defaultdict(list)
    failures = defaultdict(int)

    with open(args.path, 'rb') as handle:
        unfiltered_samples = pickle.load(handle)

    print("Scaling to 8-bit")
    unfiltered_samples = [np.uint8(sample/np.max(sample)*255) for sample in unfiltered_samples]
    
    num_cells = len(unfiltered_samples)
    print("# candidate cells: " + str(num_cells))

    print("\nTesting cropped sample cell images")
    
    load_inc = int(num_cells / 100)
    for i in range(num_cells):
        sample = unfiltered_samples[i]
        if(sample_valid(sample, args.s, failures, invalid_samples)):
            filtered_samples.append(sample)
        if i % load_inc == 0:
            print(f'Filtering: {i}/{num_cells}', end="\r")
    
    print("\nDone filtering")
    print("Samples valid: " + str(len(filtered_samples)) + "\n")
    unfiltered_samples = np.stack(unfiltered_samples, axis=0)


    print("Samples discarded:")
    [print(key, ":", failures[key]) for key in failures]

    print("\nSaving valid samples")

    if len(filtered_samples) > 0:
        if not args.i:
            print("Saving at: " + args.o)
            pickle.dump(filtered_samples, open( args.o, "wb" ))
        else:
            print("Saving at: " + args.o)
            os.mkdir(args.o)
            for key in failures.keys():
                out_file = open( args.o + "\\" + key, "wb" )
                pickle.dump(invalid_samples[key], out_file)

    elapsed = time.time() - start_time
    print("\nTotal time: " + str(elapsed))

    print('\nDataset creation completed')

if __name__=="__main__":
    main()