import numpy as np
import pickle
import os
from argparse import ArgumentParser as arg_parser

def main(raw_args=None):
    parser = arg_parser(description="Change august to december format")
    parser.add_argument("path", action="store", type=str, \
                        help="Source path for pickle containing images.")
    parser.add_argument('-l', '--list', help='delimited list input', type=str)
    parser.add_argument("-o", action="store", type=str, \
                        help="out path of dataset (default=current dir")

    # Default optional args
    parser.set_defaults(o=".\\rearranged_data")
    
    args = parser.parse_args()
    order = [int(item) for item in args.list.split(',')]
    print("Reordering dataset with order...")
    print("Order: " + str(order))
    print("Input path: " + args.path)
    print("Output path: " + args.o)

    if os.path.isfile(args.o) and args.o != ".\\rearranged_data": 
        sys.exit("output file exists... exiting")
        
    with open(args.path, "rb") as handle:
        samples = pickle.load(handle)

    rearranged_samples = []

    for sample in samples:
        rearranged_samples.append([sample[i] for i in order])

    i = 0

    pickle.dump(rearranged_samples, open( args.o, "wb" ))

if __name__=="__main__":
    main()