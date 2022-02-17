import os
import argparse
import random
import shutil
import numpy as np

parser = argparse.ArgumentParser(description='5-fold Cross Validation Image Shuffle')
parser.add_argument('data',
                    help='path to dataset')
parser.add_argument('--out', default='./split_set/',
                    help='out path for split dataset')
parser.add_argument('--k', default=5,
                    help='number of folds to split data into')

def main(raw_args=None):
    args = parser.parse_args()
    if os.path.isdir(args.out):
        shutil.rmtree(args.out)
    
    
    for k in range(args.k):
        os.makedirs(args.out + "/fold_" + str(k + 1))
    class_dirs = os.listdir(args.data)

    for class_dir in class_dirs:
        print("Processing Class " + class_dir)
        src_dir = args.data + '/' + class_dir
        files = os.listdir(src_dir)
        num_images = len(files)
        img_seq = random.sample(list(range(num_images)), k=num_images)
        img_seq = np.array_split(img_seq, args.k)
        
        for k in range(args.k):
            print("Fold " + str(k + 1) + "...")
            out_path = args.out + "/fold_" + str(k + 1) + "/" + class_dir
            os.makedirs(out_path)
            for i in img_seq[k]:
                src = src_dir + "/" + files[i]
                shutil.copy(src, out_path)
    
    return

if __name__ == '__main__':
    main()