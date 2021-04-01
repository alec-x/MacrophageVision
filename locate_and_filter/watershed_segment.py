from binarize_otsu import binarize_otsu
import numpy as np
from scipy import ndimage as ndi
from skimage.feature import peak_local_max
from skimage import filters
import skimage.morphology as morph
from skimage.segmentation import watershed

def watershed_segment(img, min_size):
    img = filters.median(img, disk(5))
    img = binarize_otsu(img, min_size)
    distance = ndi.distance_transform_edt(img)
    local_maxi = peak_local_max(distance, labels=img,
                            footprint=np.ones((3, 3)),
                            indices=False)
    markers = ndi.label(local_maxi)[0]
    labels = watershed(-distance, markers, mask=img)
    counts = np.bincount(labels.flatten())
    background_label = np.argmax(counts)
    out_arr = np.zeros_like(img)
    out_arr[labels == background_label] = 0
    out_arr[labels != background_label] = 1 
    out_arr = morph.remove_small_objects(out_arr, min_size)

    return out_arr, labels