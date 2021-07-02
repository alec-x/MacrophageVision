from math import sqrt
from skimage import data
from skimage.feature import blob_dog, blob_log, blob_doh
from skimage.color import rgb2gray
import cv2
import matplotlib.pyplot as plt


image = cv2.imread('data/external/20x_100ul007_RGB_DAPI_20X.tif')
image_gray = rgb2gray(image)

#blobs_log = blob_log(image_gray, max_sigma=20, num_sigma=7, threshold=.03)
#blobs_log[:, 2] = blobs_log[:, 2] * sqrt(2)

blobs_log_2 = blob_log(image_gray, max_sigma=20, num_sigma=7, threshold=0.05)
blobs_log_2[:, 2] = blobs_log_2[:, 2] * sqrt(2)

blobs_log_3 = blob_log(image_gray, max_sigma=20, num_sigma=7, threshold=0.7)
blobs_log_3[:, 2] = blobs_log_3[:, 2] * sqrt(2)


"""
blobs_list = [blobs_log, blobs_log_2, blobs_log_3]
colors = ['yellow', 'green', 'red']
titles = ['Thresh=0.03', 'Thresh=0.05', 'Thresh=0.07']
fig, axes = plt.subplots(1, 3, figsize=(9, 3), sharex=True, sharey=True)
"""
blobs_list = [blobs_log_2, blobs_log_3]
colors = ['yellow', 'green']
titles = ['Blob Detection', 'Input']
fig, axes = plt.subplots(1, 2, figsize=(18, 6), sharex=True, sharey=True)

ax = axes.ravel()

sequence = zip(blobs_list, colors, titles)
for idx, (blobs, color, title) in enumerate(sequence):
    ax[idx].set_title(title)
    ax[idx].imshow(image_gray, cmap='gray')
    for blob in blobs:
        y, x, r = blob
        c = plt.Circle((x, y), r, color=color, linewidth=0.5, fill=False)
        ax[idx].add_patch(c)
    ax[idx].set_axis_off()

plt.tight_layout()
plt.show()