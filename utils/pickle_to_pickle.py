import os
import cv2
import numpy
import pickle

data = pickle.load(open("kerryn_data_scripts\\kerryn_all_filtered.pickle", "rb"))
out_imgs = []
print(len(data))
print("aggregating...")

for i in range(len(data)):
    img = [data[i][0], data[i][1], data[i][2], data[i][3]]
    out_imgs.append(img)
    
print("saving to pickle_to_pickle.pickle")
pickle.dump(out_imgs, open( "pickle_to_pickle.pickle", "wb" ) )