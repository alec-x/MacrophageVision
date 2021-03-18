import numpy as np
from skimage.filters import threshold_otsu
import skimage.morphology as morph

def binarize_otsu(img, min_obj_size):
    arr = np.copy(img)
    threshold = threshold_otsu(arr)
    arr[arr < threshold] = 0
    arr = arr.astype(bool)
    arr = morph.remove_small_objects(arr, min_obj_size)
    return arr