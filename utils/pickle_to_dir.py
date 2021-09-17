from argparse import ArgumentParser as arg_parser
import pickle
from PIL import Image
import math
import numpy as np
import os
import sys

def main(raw_args=None):
    parser = arg_parser(description="Create dataset from cell images in pickle.")
    parser.add_argument("path", action="store", type=str, \
                        help="Source path for pickle.")
    parser.add_argument("-o", action="store", type=str, \
                        help="out path of dir (default=current dir")

    # Default optional args
    parser.set_defaults(o=".\\data")
    args = parser.parse_args()

    if os.path.exists(args.o): 
        sys.exit("output path exists... exiting")
    
    data = None
    with open(args.path, 'rb') as handle:
        data = pickle.load(handle)
    
    images = data["images"]
    channel_order = data["channels"][0]
    print("Saving at: " + args.o)
    sig_figs = math.floor(math.log10(len(data))) + 1
    os.makedirs(args.o)
    for channel in channel_order:
        os.mkdir(args.o + "\\" + channel)

    load_inc = int(len(images) / 100)

    for i, sample in enumerate(images):
        if i % load_inc == 0:
            print(f'Saving: {i}/{len(images)}', end="\r")
            
        
        curr_name = str(i).rjust(sig_figs,'0')
        for j, chan in enumerate(channel_order):
            tmp_img = Image.fromarray(np.array(sample[j])).convert("L")
            tmp_img.save(args.o + "\\"+ chan +"\\" + curr_name + chan + '.png')

    print("\n image conversion complete")

if __name__=="__main__":
    main()