from __future__ import print_function, division
from argparse import ArgumentParser as arg_parser
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle
import os
import shutil
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
    
    parser.set_defaults(o=".\\default_dataset")
    
    args = parser.parse_args()

    print("Input path: " + args.path)
    print("Output path: " + args.o)

    data_file = args.path + '\\data.pickle'

    if not os.path.isfile(data_file):
        sys.exit("data pickle not in directory... exiting")
    if os.path.isdir(args.o): 
        sys.exit("output path exists... exiting")    

    print("\nLoading pickle")
    raw_data = pickle.load(open(data_file, "rb"))

    data_selections = []

    print("\nApplying selection criteria")
    selection = raw_data['cd80_measure'] > 50
    data_selections.append(raw_data.loc[selection])

    selection = raw_data['cd80_measure'] < 10
    data_selections.append(raw_data.loc[selection])
    
    data_lists = []
    for selected in data_selections:
        list_1 = selected["bf_path"].tolist()
        list_2 = selected["mito_path"].tolist()
        data_lists.append(tuple(zip(list_1, list_2)))

    print("\nSaving to output directory")
    os.mkdir(args.o)

    class_name = 0

    for data_list in data_lists:
        base_name =  args.o + "\\" + str(class_name)
        os.mkdir(base_name)

        bf_name = base_name + "\\" + "bf"
        mito_name = base_name + "\\" + "mito"
        os.mkdir(bf_name)
        os.mkdir(mito_name)

        class_name += 1

        for sample in data_list:
            shutil.copy(args.path + "\\" + sample[0], bf_name)
            shutil.copy(args.path + "\\" + sample[1], mito_name)
    
if __name__=="__main__":
    main()