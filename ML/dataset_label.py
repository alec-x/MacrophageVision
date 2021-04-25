from __future__ import print_function, division
from argparse import ArgumentParser as arg_parser
import csv
import pandas as pd
import pickle
import os
import shutil
import sys
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

    print("\nApplying selection criteria")
    criterion = []
    criterion.append(raw_data['cd80_measure'] > 60) 
    criterion.append(raw_data['cd80_measure'] < 15)

    data_selections = []
    for i, criteria in enumerate(criterion):
        selected = raw_data.loc[criteria][["bf_path", "mito_path"]]
        selected["label"] = i
        data_selections.append(selected)
    
    processed = pd.concat(data_selections)

    print("\nSaving to output directory")
    
    os.mkdir(args.o)

    class_name = 0
            
    with open(args.o + '\\labels.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',', 
                           quotechar='|', quoting=csv.QUOTE_MINIMAL) 
        writer.writerow(["bf_path", "mito_path", "label"])
        for row in processed.itertuples(index=False):
            
            in_bf    = args.path + "\\" + row[0]
            in_mito  = args.path + "\\" + row[1]
            bf_file = row[0].split("\\")[-1]
            mito_file =row[1].split("\\")[-1]
            out_bf   = args.o + "\\" + bf_file
            out_mito = args.o + "\\" + mito_file

            shutil.copy(in_bf, out_bf)
            shutil.copy(in_mito, out_mito)
            writer.writerow([bf_file, mito_file, row[2]])
            
if __name__=="__main__":
    main()