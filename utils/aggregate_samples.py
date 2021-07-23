import pickle
import numpy
import os
from matplotlib import pyplot as plt

def get_all_files(path, extension):
    pathlist = []
    #Look in all subfolders
    for root, _, files in os.walk(path):
        for name in files:
            #If the file is a tif file
            if name.endswith('.' + str(extension)):
                p = os.path.join(root, name)
                pathlist.append(p)

    return pathlist

paths = [("alveolar", r"data\raw\alveolar_autof\certain")]
paths.append(("marrow", r"data\raw\bm_autof\certain"))


labels = [get_all_files(path[1], "pickle") for path in paths]
#[[path_1,path_2,path_3,path_4],[...,...,...,...]]

datas = [[pickle.load(open(file, "rb")) for file in file_list] for file_list in labels]

agg_data = [[] for _ in range(len(datas))]

print("aggregating data")
for i, data in enumerate(datas):
    for imgs in data:
        agg_data[i].extend(imgs)
        
for data in zip(paths, agg_data):
    print(f"{data[0][0]}, # total: {len(data[1])}")
    