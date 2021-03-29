import os
import pickle
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import filedialog
import numpy as np

"""
Main Window Initialization
"""
top = tk.Tk()
top.title("Manual Macrophage Labeler")

"""
State Vars 
"""
image_list = []
curr_imgs = [None]*4
curr_num = 0
curr_num_var = tk.StringVar()
curr_num_var.set("0")
total_num = 0
total_num_var = tk.StringVar()
total_num_var.set("0")
in_path_var = tk.StringVar()
out_path_var = tk.StringVar()

"""
Widgets
"""
in_path_lbl = tk.Label(top, text="Input images directory:")
in_path_txt = tk.Entry(top, textvariable=in_path_var)
load_btn = tk.Button(top, text="Load data")

can = tk.Canvas(top, width=800, height=200)
can.image = None

curr_txt = tk.Entry(top, text=curr_num_var)
curr_slash = tk.Label(top, text="out of")
curr_lbl_2 = tk.Entry(top, text=total_num_var)
jump_btn = tk.Button(top, text="jump to")
left_btn = tk.Button(top, text="<<")
right_btn = tk.Button(top, text=">>")

"""
Functions
"""
def browse_file(frame, textbox):
    path = filedialog.askopenfilename(parent=frame, title="Please select the macrophage pickle")
    if len(path) > 0:
        textbox.delete(0, "end")
        textbox.insert(0,path)
        print("Selected path: " + path)
        return 0
    else:
        return 1

def change_img(num, image_list):
    global curr_imgs
    arr_0 = np.zeros_like(image_list[num][0])
    
    curr_imgs[0] = Image.fromarray(image_list[num][0])
    curr_imgs[1] = Image.fromarray(np.stack((arr_0, image_list[num][1], arr_0), axis=2))
    temp2 = image_list[num][2]

    curr_imgs[2] = Image.fromarray(np.stack((arr_0, arr_0, temp2), axis=2))
    temp3 = image_list[num][3]

    curr_imgs[3] = Image.fromarray(np.stack((temp3, arr_0, arr_0), axis=2))

    can.image0 = ImageTk.PhotoImage(curr_imgs[0].resize((192,192)))
    can.image1 = ImageTk.PhotoImage(curr_imgs[1].resize((192,192)))
    can.image2 = ImageTk.PhotoImage(curr_imgs[2].resize((192,192)))
    can.image3 = ImageTk.PhotoImage(curr_imgs[3].resize((192,192)))
    
    can.delete("all")
    can.create_image((0,0), image=can.image0, anchor="nw")
    can.create_image((200,0), image=can.image1, anchor="nw")
    can.create_image((400,0), image=can.image2, anchor="nw")
    can.create_image((600,0), image=can.image3, anchor="nw")
    
    return 0

def load_in(path, data_list):
    with open(path, 'rb') as handle:
        data_list = pickle.load(handle)
    global total_num
    global total_num_var
    global image_list
    total_num = len(data_list)
    total_num_var.set(str(total_num))
    image_list = data_list
    change_img(0, image_list)

def jump_num(num):
    global curr_num
    global curr_num_var    
    global image_list
    if num < 0:
        num = 0
    if num > int(total_num_var.get()):
        num = int(total_num_var.get())
    curr_num = num
    curr_num_var.set(num)
    change_img(num, image_list)
    change_img(num, image_list)
    
"""
Button Press Bindings
"""
curr_lbl_2.bind("<Key>", lambda e: "break")
in_path_txt.bind("<ButtonPress-1>", lambda event: browse_file(top, in_path_txt))
load_btn.bind("<ButtonPress-1>", lambda event: load_in(in_path_var.get(), image_list))
top.bind('<Return>', lambda event: jump_num(int(curr_num_var.get())))
left_btn.bind("<ButtonPress-1>", lambda event: jump_num(int(curr_num_var.get())-1))
right_btn.bind("<ButtonPress-1>", lambda event: jump_num(int(curr_num_var.get())+1))

"""
Grid Layout
"""
x_pad = 4
y_pad = 4

in_path_lbl.grid(row=0, column=0, sticky='E', padx=x_pad, pady=y_pad)
in_path_txt.grid(row=0, column=1, columnspan=5,padx=x_pad, pady=y_pad, sticky='WE')
load_btn.grid(row=0, column=6, padx=x_pad, pady=y_pad, sticky='W')

can.grid(row=2, columnspan=7, padx=x_pad, pady=y_pad)

left_btn.grid(row=4, column=1, sticky='E', padx=x_pad, pady=y_pad)
curr_txt.grid(row=4, column=2, sticky='WE', padx=x_pad, pady=y_pad)
right_btn.grid(row=4, column=3, sticky='W', padx=x_pad, pady=y_pad)
curr_slash.grid(row=4, column=4, sticky='E', padx=x_pad, pady=y_pad)
curr_lbl_2.grid(row=4, column=5, sticky='W', padx=x_pad, pady=y_pad)



#jump_btn.grid(row=4, column=4, sticky='NSEW', padx=x_pad, pady=y_pad)
top.mainloop()

