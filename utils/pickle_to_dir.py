from argparse import ArgumentParser as arg_parser
import pickle
from PIL import Image
import math
import numpy as np
import os
import sys

def main(raw_args=None):
    parser = arg_parser(description="Create dataset from cell images in dir. \
        \n Input requires a directory with images under directories with \
        corresponding cell name")
    parser.add_argument("path", action="store", type=str, \
                        help="Source path for dir containing pickle.")
    parser.add_argument("-o", action="store", type=str, \
                        help="out path of dataset (default=current dir")

    # Default optional args
    parser.set_defaults(o=".\\data")
    args = parser.parse_args()

    if os.path.exists(args.o): 
        sys.exit("output path exists... exiting")
    
    data = None
    with open(args.path, 'rb') as handle:
        data = pickle.load(handle)
    
    print("Saving at: " + args.o)
    sig_figs = math.floor(math.log10(len(data))) + 1

    os.makedirs(args.o)
    os.mkdir(args.o + "\\" + "BF")
    os.mkdir(args.o + "\\" + "mito")
    os.mkdir(args.o + "\\" + "cd80") #M1, blue
    os.mkdir(args.o + "\\" + "cd206") #M2, red
    
    i = 0
    load_inc = int(len(data) / 100)

    for sample in data:      
        """
        if i % load_inc == 0:
            print(f'Saving: {i}/{len(data)}', end="\r")
        """
        tmp_img = Image.fromarray(sample[0])
        curr_name = str(i).rjust(sig_figs,'0')
        
        tmp_img.save(args.o + "\\BF\\" + curr_name + 'BF.png')
        
        tmp_img = Image.fromarray(sample[1])
        tmp_img.save(args.o + "\\mito\\" + curr_name + 'mito.png')
        
        tmp_img = Image.fromarray(sample[2])
        tmp_img.save(args.o + "\\cd80\\" + curr_name + 'cd80.png')

        tmp_img = Image.fromarray(sample[3])
        tmp_img.save(args.o + "\\cd206\\" + curr_name + 'cd206.png')
        
        i += 1

if __name__=="__main__":
    main()