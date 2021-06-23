import os, sys
import cv2
import os
import glob

os.chdir("D:\\data\\processed\\dataset_1\\bone_marrow_stained\\non-blurred_bf")
os.makedirs("blurred")
i = 0
for file in glob.glob("*.png"):
    i += 1
    img = cv2.imread(file, cv2.IMREAD_GRAYSCALE )
    blur = cv2.GaussianBlur(img,(5,5),2)
    filename = "blurred"+file
    if not cv2.imwrite("blurred" + "\\" + file,blur): print("failed")
    if i % 100 == 0:
        print("processed: " + str(i), end="\r")

print("\n finished")