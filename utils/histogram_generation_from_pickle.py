import pickle
import numpy as np
import pandas
import cv2
from matplotlib import pyplot as plt
from PIL import Image

data = pickle.load(open("C:\\Users\\Alec\\Documents\\Source\\Repos\\MDLMacrophageVision\\data\\kerryn_all_manually_filtered_measured.pickle", "rb"))

bins = list(range(-10,271,5))
f, axarr = plt.subplots(1,2) 
f.set_figheight(5)
f.set_figwidth(14)
axarr[0].hist(data["CD80_diff"], bins = bins)
axarr[1].hist(data["CD206_diff"], bins = bins)
axarr[0].set_title("CD80 top 20% px brightness")
axarr[1].set_title("CD206 top 20% px brightness")
axarr[0].set_xlabel('Top 20% pixel intensity in cell')
axarr[1].set_xlabel('Top 20% pixel intensity in cell')
axarr[0].set_ylabel('Count')
axarr[1].set_ylabel('Count')
plt.tight_layout()
plt.savefig("./hist.png")
