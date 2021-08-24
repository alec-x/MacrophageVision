import pickle
import numpy as np
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

out_path_base = ".\data\processed"
base = r'D:\data\processed\autof_july_2021'

paths = [("alveolar", base + r'\alveolar_autof\certain'),
         ("marrow", base + r'\bm_autof\certain'),
         ("monocyte", base + r'\monocytes_autof\certain')]

# Alveolar autof: ["green", "red", "blue", "bf"]   
# BM autof: ["green", "blue", "red", "bf"]   
# Monocytes autof: ["green", "red", "blue", "bf"]  
channel_order = [["green", "red", "blue", "bf"],
                 ["green", "blue", "red", "bf"]]

labels = [get_all_files(path[1], "pickle") for path in paths]
#[[path_1,path_2,path_3,path_4],[...,...,...,...]]

datas = [[pickle.load(open(file, "rb")) for file in file_list] for file_list in labels]

agg_data = [[] for _ in range(len(datas))]

print("aggregating data")
for i, data in enumerate(datas):
    for imgs in data:
        for img in imgs:
            if img[0].shape == (96,96):
                stacked_img = np.array(img)
                agg_data[i].append(stacked_img)
print("\nSaving to pickle")
for i, data in enumerate(zip(paths, agg_data)):

    num_samples = len(data[1])
    img_shape = data[1][1].shape
    print(f"{data[0][0]}, # total: {num_samples}")
    arr_shape = (num_samples,) + img_shape
    arr_data = np.zeros(arr_shape)
    arr_labels = np.zeros(num_samples) + i
    for j in range(num_samples):
        arr_data[j, :] = data[1][j]
    #
    output = {}
    output["labels"] = arr_labels
    output["channels"] = channel_order
    output["images"] = arr_data
    out_path = out_path_base + '\\' + data[0][0] + ".pickle"
    pickle.dump(output, open(out_path, "wb" ))
    print(f"Saved {data[0][0]} to {out_path}")