import os
from os import listdir
from PIL import Image

dir_path = r"D:\data\processed\dataset_1\bone_marrow_stained\bf"

i = 0
for filename in listdir(dir_path):
    if i % 100 == 0:
        print(i)
    i += 1
    if filename.endswith('.png'):
        try:
            img = Image.open(dir_path+"\\"+filename) # open the image file
            img.verify() # verify that it is, in fact an image
        except (IOError, SyntaxError) as e:
            print('Bad file:', filename)
            #os.remove(base_dir+"\\"+filename) (Maybe)