# https://stackoverflow.com/questions/22159160/python-calculate-histogram-of-image
import matplotlib.pyplot as plt
import cv2
import numpy as np
from skimage.filters import threshold_triangle
import skimage.morphology as morph
from skimage.filters import try_all_threshold

def binarize_triangle(img, offset):
    arr = np.copy(img)
    threshold = threshold_triangle(arr)

    return threshold

im = cv2.imread(r'data/external/1012_RGB_EGFP_20X.tif')

im = im[:,:,1]

thresh = binarize_triangle(im, 0)


#cv2.imshow("asd", im)
#plt.show()


# flatten to 1D array
vals = im.flatten()
# plot histogram with 255 bins
b, bins, patches = plt.hist(vals, 25)
plt.xlim([25,255])
plt.show()