from __future__ import print_function, division
from argparse import ArgumentParser as arg_parser
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle
import os
from skimage import io, transform
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import warnings

warnings.filterwarnings("ignore")

def main(raw_args=None):
    parser = arg_parser(description="Create dataset from cell images in dir. \
        \n Input requires a directory with images under directories with \
        corresponding cell name")
    parser.add_argument("path", action="store", type=str, \
                        help="Source path for dir containing raw images.")
    parser.add_argument("-o", action="store", type=str, \
                        help="out path of dataset (default=current dir")
    args = parser.parse_args()
    parser.set_defaults(o=".\\default_dataset")

    print("Input path: " + args.path)
    print("Output path: " + args.o)

    data_file = args.path + '\\data.pickle'

    if not os.path.isfile(data_file):
        sys.exit("data pickle not in directory... exiting")
    if os.path.exists(args.o): 
        sys.exit("output path exists... exiting")    

    print("\nLoading pickle")
    raw_data = pickle.load(open(data_file, "rb"))

    selection = raw_data['cd80_measure'] > 50
    selected_data = raw_data.loc[selection]
    print(selected_data)

if __name__=="__main__":
    main()