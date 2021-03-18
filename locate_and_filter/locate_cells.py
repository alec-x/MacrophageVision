# Create dataset
# Alec Xu
# Partial code from Samuel Berryman
# Multi-scale Design Lab
# University of British Columbia
# 2020
from argparse import ArgumentParser as arg_parser
from binarize_otsu import binarize_otsu
import concurrent.futures
import numpy as np
import os, os.path
import pickle
from PIL import Image
import scipy.ndimage as ndimage
from skimage.filters import threshold_otsu
import skimage.morphology as morph
from skimage.segmentation import clear_border
import sys
import time

# Return list of list of channels. Each list of a channel is a frame. 
# Each channel is represented by a numpy array
def read_tiff(path, channels, offset=0):
    imagestack = []
    curr_stack = []

    img = Image.open(path)
    
    i = 0 # Number of frames visited starting at offset
    channel = 0 # the current channel
    
    while(1):
        try:
            image = img
            image.seek(offset+i) # goto frame with index

            image = np.array(image) # convert frame to numpy array
            # convert to uint8 range, then type
            image = image.astype(np.int32)* 255/image.astype(np.int32).max()
            image = np.array(image, dtype=np.uint8) # convert to uint8
            curr_stack.append(image) # Add channel to current stack
            channel += 1
            # If current stack has all channels add to image stack
            if(channel >= channels): 
                imagestack.append(curr_stack)
                curr_stack = []
                channel = 0
            i += 1
        except EOFError:
            # end of image sequence. End loop
            break

    return imagestack

# Create a list of all of the images/labels in the training folder
def get_all_files(path, extension):
    pathlist = []
    labels = []
    #Look in all subfolders
    for root, _, files in os.walk(path):
        for name in files:
            #If the file is a tif file
            if name.endswith('.' + str(extension)):
                p = os.path.join(root, name)
                pathlist.append(p)
                #Append the label from the folder it is in
                labels.append(os.path.basename(os.path.dirname(p)))
    return pathlist, labels

# Accept numpy array of cell stain, and search box size.
# Returns a list of cell centre indices in tuple (x,y) form
def locate_cells(stain):
    img = stain.copy() # Create local copy of data
    threshold = threshold_otsu(img)
    img[img < threshold] = 0
    img = img.astype(bool)
    img = clear_border(img)
    img = morph.remove_small_objects(img, min_size=150)
    labels, num_objs = ndimage.label(img)
    obj_list = list(range(1,num_objs+1)) # required for center_of_mass
    locations = ndimage.measurements.center_of_mass(stain, labels, obj_list)
    return locations, img, labels

def process_cells_worker(path, box_size):
    frame_stack = read_tiff(path,4)
    x_max, y_max = frame_stack[0][0].shape
    output_stack = []
    for frame in frame_stack:
        mito_stain = frame[2] # mitochondria stain always on 2nd frame
        locations, binary, labels = locate_cells(mito_stain)
        curr_frame = []
        
        for location in locations:
            x1 = int(location[0] - box_size/2)
            x2 = int(location[0] + box_size/2)
            y1 = int(location[1] - box_size/2)
            y2 = int(location[1] + box_size/2)
            if not (x1 < 0 or y1 < 0): 
                crop = [img[x1:x2, y1:y2] for img in frame]
                num_objs = len(np.unique(labels[x1:x2, y1:y2])) - 1
                if num_objs == 1 and crop[0].shape == (box_size, box_size):
                    curr_frame.append(crop)
        if len(curr_frame) > 0:
            output_stack.extend(np.stack(curr_frame, axis=0))
    return output_stack

def main(raw_args=None):
    parser = arg_parser(description="Create pickle of candidate cells from cell images in dir. \
        \n Input requires a directory with images under directories with corresponding cell name")
    parser.add_argument("path", action="store", type=str, \
                        help="Source path for dir containing raw images.")
    parser.add_argument("-s", action="store", type=int, \
                        help="size of training image in px (default=96")
    parser.add_argument("-o", action="store", type=str, \
                        help="out path of dataset (default=current dir")
    parser.add_argument("-t", action="store", type=int, \
                        help="num threads (default 8)")
    parser.set_defaults(s=96, o=".\\data", t=8)
    args = parser.parse_args(raw_args)

    print("Creating pickle with params...") 
    print("Image size: " + str(args.s) + "px")
    print("Input path: " + args.path)
    print("Output path: " + args.o)

    if os.path.isfile(args.o) and args.o != ".\\data": 
        sys.exit("output file exists... exiting")
            
    # Get a list of image paths of cells with labels
    print("\nLocating all images in directory")
    start_time = time.time()
    paths, labels = get_all_files(args.path, 'tif')

    # Create dict of lists to hold each label type
    unfiltered_samples = []


    num_imgs = 1
    tot_imgs = len(paths)
    print("Locating and isolating cells in images")
    print("Found " + str(tot_imgs) + " images")
    print("Initializing multithreading")
    # If become CPU bound, use ProcessPool instead
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.t) as executor:
        futures = []
        for path in paths:
            futures.append(executor.submit( \
                process_cells_worker,path=path, box_size=args.s))

        for future in concurrent.futures.as_completed(futures):
            try:
                stacks = future.result()
            except Exception as e:
                print("error while getting cell samples from images")
                print(e)
            else:
                unfiltered_samples.extend(stacks)
                print(f'Image {num_imgs}/{tot_imgs}', end="\r")
                num_imgs += 1
    print("\n")

    print("Saving at: " + path_full)
    unfiltered_samples = np.stack(unfiltered_samples, axis=0)
    pickle.dump(unfiltered_samples, open(path_full, "wb" ))

    elapsed = time.time() - start_time
    print("\nTotal time: " + str(elapsed))

    print('\nDataset creation completed')

if __name__=="__main__":
    main()