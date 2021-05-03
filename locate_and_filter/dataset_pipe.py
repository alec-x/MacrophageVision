from argparse import ArgumentParser as arg_parser
from filter_basic import main as filter_basic
from locate_cells import main as locate_cells
from thresholding import main as thresholding
from os import remove
import uuid


def main(raw_args=None):
    parser = arg_parser(description="Run pytorch dataset creation pipeline")
    parser.add_argument("path", action="store", type=str, \
                        help="Source path for dir containing raw tiffs.")
    parser.add_argument("-s", action="store", type=int, \
                        help="size of training image in px")
    parser.add_argument("-o", action="store", type=str, \
                        help="out path of dataset")
    parser.add_argument("-c", action="store", type=int, \
                        help="clear interim files (default=true)")
    suffix = str(uuid.uuid4())
    parser.set_defaults(s=96, o=".\\data_" + suffix, t=8)
    args = parser.parse_args(raw_args)

    located_path = "data\\interim\\located_" + suffix
    print("\nlocating cells")
    print("=================================================================")
    locate_cells([args.path, "-o", located_path])
    print("=================================================================")

    filtered_path = "data\\interim\\filtered_" + suffix 
    print("\nPerforming basic filtering")
    print("=================================================================")
    filter_basic([located_path, "-o", filtered_path])
    print("=================================================================")

    thresholded_path = "data\\processed\\thresholded_" + suffix 
    print("\nPerforming basic filtering")
    print("=================================================================")
    thresholding([filtered_path, "-o", thresholded_path])
    print("=================================================================")

    
    if args.c:
        print("\nRemoving intermediary files")
        remove(located_path)
        remove(filtered_path)

if __name__=="__main__":
    main()    