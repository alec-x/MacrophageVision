import numpy as np
def colourize(img, colour):
    arr_0 = np.zeros_like(img)
    return np.stack((arr_0, arr_0, img), axis=2)