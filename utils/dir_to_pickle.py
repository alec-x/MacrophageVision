import os
import cv2
import numpy
import pickle

imgs = {}
imgs["bf"] = []
imgs["mito"] = []
imgs["cd80"] = []
imgs["cd206"] = []

out_imgs = []
i = 0
for root, dirs, files in os.walk(r"D:\\output"):
    for file in files:
        if file.endswith(".png"):
            img = cv2.imread(os.path.join(root, file), cv2.IMREAD_GRAYSCALE)
            if i % 1000 == 0:
                print(i)
            if img is not None:
                if os.path.basename(root) == "BF":
                    imgs["bf"].append(img)
                elif os.path.basename(root) == "Mito":
                    imgs["mito"].append(img)
                elif os.path.basename(root) == "CD80":
                    imgs["cd80"].append(img)
                elif os.path.basename(root) == "CD206":
                    imgs["cd206"].append(img)                    
            i += 1


print("aggregating...")

for i in range(len(imgs["bf"])):
    img = [imgs["bf"][i], imgs["mito"][i], imgs["cd80"][i], imgs["cd206"][i]]
    out_imgs.append(img)
    
print("saving to dir_to_pickle.pickle")
pickle.dump(out_imgs, open( "dir_to_pickle.pickle", "wb" ) )