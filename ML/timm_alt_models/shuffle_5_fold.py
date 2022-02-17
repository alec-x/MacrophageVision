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
        print("deleting existing directory at out path")
        shutil.rmtree(args.out)
    
    class_dirs = os.listdir(args.data)
    
    img_seq = {}
    for class_dir in class_dirs:
        img_seq[class_dir] = {}
        src_dir = args.data + '/' + class_dir
        files = os.listdir(src_dir)
        num_img = len(files)
        sequence = random.sample(list(range(num_img)), k=num_img) 
        
        for k in range(args.k):
            img_seq[class_dir][k] = {}
            folds = list(range(args.k))
            folds.remove(k)
            split_seq = np.array_split(sequence, args.k) 

            train = [list(split_seq[fold]) for fold in folds]
            train = sum(train, [])
            test = list(split_seq[k])

            img_seq[class_dir][k]["train"] = train
            img_seq[class_dir][k]["test"] = test

    for k in range(args.k):
        
        out_path = args.out + "/fold_" + str(k + 1)
        os.makedirs(out_path + "/train")
        os.makedirs(out_path + "/test")

        for class_dir in class_dirs:
            print("Processing Fold " + str(k + 1) + " " + class_dir,end='\r')
            src_dir = args.data + '/' + class_dir
            os.makedirs(out_path + "/train/" + class_dir)
            os.makedirs(out_path + "/test/" + class_dir)
            files = os.listdir(src_dir)
                
            for i in img_seq[class_dir][k]["train"]:
                src = src_dir + "/" + files[i]
                shutil.copy(src, out_path + "/train/" + class_dir)
            for i in img_seq[class_dir][k]["test"]:
                src = src_dir + "/" + files[i]
                shutil.copy(src, out_path + "/test/" + class_dir)
    return

if __name__ == '__main__':
    main()