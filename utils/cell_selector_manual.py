# https://stackoverflow.com/questions/17466561/best-way-to-structure-a-tkinter-application
from posixpath import basename
import numpy as np
from PIL import ImageTk, Image
import tkinter as tk
from tkinter import filedialog
import pickle
import ntpath

XPAD = 8
YPAD = 4
IMAGE_SIZE = 96
class PathBar(tk.Frame):
    def __browse_file(self, textbox):
        path = filedialog.askopenfilename(parent=self, title="Please select the nanowell image")
        if len(path) > 0:
            textbox.delete(0, "end")
            textbox.insert(0,path)
            print("Selected path: " + path)
            self.parent.display.can.focus_set()
            return 0
        else:
            self.parent.display.can.focus_set()
            return 1
        
    def __init__(self, parent, *args, **kwargs):
        tk.Frame.__init__(self, parent, *args, **kwargs)
        self.parent = parent
       
        in_path_lbl = tk.Label(self, text="Image path:")
        in_path_txt = tk.Entry(self, textvariable=parent.in_path)

        in_path_lbl.pack(side="left", padx=XPAD, pady=YPAD)
        in_path_txt.pack(side="left", fill="x", expand=True, padx=XPAD, pady=YPAD)

        in_path_txt.bind("<ButtonPress-1>", lambda event: self.__browse_file(in_path_txt))

class SaveBar(tk.Frame):
    def __browse_folder(self, textbox):
        path = filedialog.askdirectory(parent=self, title="Please select output directory")
        if len(path) > 0:
            textbox.delete(0, "end")
            textbox.insert(0,path)
            print("Selected directory: " + path)
            return 0
        else:
    
            return 1

    def __init__(self, parent, *args, **kwargs):
        tk.Frame.__init__(self, parent, *args, **kwargs)
        self.parent = parent
       
        self.out_path_lbl = tk.Label(self, text="Save dir:      ")
        self.out_path_txt = tk.Entry(self, textvariable=parent.out_path)
        self.save_btn = tk.Button(self, text="Save", width=20)
        
        self.out_path_lbl.pack(side="left", padx=XPAD, pady=YPAD)
        self.out_path_txt.pack(side="left", fill="x", expand=True, padx=XPAD, pady=YPAD)
        self.save_btn.pack(side="left", padx=XPAD, pady=YPAD)

        self.out_path_txt.bind("<ButtonPress-1>", lambda event: self.__browse_folder(self.out_path_txt))
        

class LayerBar(tk.Frame):
    def __init__(self, parent, *args, **kwargs):
        tk.Frame.__init__(self, parent, *args, **kwargs)
        self.parent = parent    
        
        layer_lbl = tk.Label(self, text="Current Layer:")
        layer_drop = tk.OptionMenu(self, parent.curr_layer, *parent.layer)
        
        layer_lbl.pack(side="left", padx=XPAD, pady=YPAD)
        layer_drop.pack(side="left", padx=XPAD, pady=YPAD)

class Display(tk.Frame):
    def motion(self, event):
        self.can.x, self.can.y = event.x, event.y

    def draw_markers(self, certain_stack, uncertain_stack):
        canv = self.can
        canv.delete("all")
        canv.create_image((0,0), image=canv.image, anchor="nw")
        
        horz_ratio = self.horz_ratio.get()
        vert_ratio = self.vert_ratio.get()

        for location in certain_stack:
            x1 = location[0] - IMAGE_SIZE * vert_ratio / 2
            y1 = location[1] - IMAGE_SIZE * horz_ratio / 2
            x2 = location[0] + IMAGE_SIZE * vert_ratio / 2
            y2 = location[1] + IMAGE_SIZE * horz_ratio / 2
            canv.create_rectangle(x1, y1, x2, y2,outline="green")

        for location in uncertain_stack:
            x1 = location[0] - IMAGE_SIZE * vert_ratio / 2
            y1 = location[1] - IMAGE_SIZE * horz_ratio / 2
            x2 = location[0] + IMAGE_SIZE * vert_ratio / 2
            y2 = location[1] + IMAGE_SIZE * horz_ratio / 2
            canv.create_rectangle(x1, y1, x2, y2,outline="orange")
            
    def load_img(self, path, stack):
        
        imagestack = []
        curr_stack = []

        img = Image.open(path)
        
        i = 0 # Number of frames visited starting at offset
        channel = 0 # the current channel
        offset = 0
        while(1):
            try:
                image = img
                image.seek(offset+i) # goto frame with index

                image = np.array(image) # convert frame to numpy array
                # convert to uint8 range, then type
                image = image.astype(np.int32)* 255/image.astype(np.int32).max()
                image = np.array(image, dtype=np.uint8) # convert to uint8
                curr_stack.append(image) # Add channel to current stack
                channel += 1
                if(channel >= 4): 
                    imagestack.append(curr_stack)
                    curr_stack = []
                    channel = 0
                i += 1
            except EOFError:
                break
        stack.clear()
        stack.extend(imagestack)
        self.render_image(stack[0][0])
        self.parent.display.can.focus_set()

    def render_image(self, img):
        canv = self.can
        real_height, real_width = img.shape
        
        disp_width = canv.winfo_width()
        disp_height = canv.winfo_height()

        img =  Image.fromarray(img)
        img = img.resize((disp_width, disp_height))
        canv.delete("all")
        canv.image = ImageTk.PhotoImage(img)
        canv.create_image((0,0), image=canv.image, anchor="nw")
        ratio_vert = disp_width / real_width
        ratio_horz = disp_height / real_height
        self.horz_ratio.set(ratio_horz)
        self.vert_ratio.set(ratio_vert)

    def __init__(self, parent, *args, **kwargs):
        tk.Frame.__init__(self, parent, *args, **kwargs)
        self.horz_ratio = tk.DoubleVar(self)
        self.vert_ratio = tk.DoubleVar(self)

        self.parent = parent
        self.can = tk.Canvas(self, bg='white')
        self.can.image = None
        self.can.pack(expand=True, fill="both")

        self.can.bind('<Motion>', lambda event: self.motion(event))

class PageBar(tk.Frame):
    def __init__(self, parent, *args, **kwargs):
        tk.Frame.__init__(self, parent, *args, **kwargs)
        self.parent = parent    
        
        self.page_lbl = tk.Label(self, text="Page:")
        self.page_num_lbl = tk.Label(self, textvariable=parent.curr_page)
        self.left_btn = tk.Button(self, text="<-", width=10)
        self.right_btn = tk.Button(self, text="->", width=10)
        self.samples_lbl = tk.Label(self, text="# Samples Certain:")
        self.samples_num_lbl = tk.Label(self, textvariable=parent.num_samples_certain)
        self.samples_lbl_2 = tk.Label(self, text="# Samples Uncertain:")
        self.samples_num_lbl_2 = tk.Label(self, textvariable=parent.num_samples_uncertain)
        
        self.left_btn.pack(side="left", padx=XPAD, pady=YPAD)
        self.page_lbl.pack(side="left", padx=XPAD, pady=YPAD)
        self.page_num_lbl.pack(side="left", padx=XPAD, pady=YPAD)
        self.right_btn.pack(side="left", padx=XPAD, pady=YPAD)
        self.samples_num_lbl.pack(side="right", padx=XPAD, pady=YPAD)
        self.samples_lbl.pack(side="right", padx=XPAD, pady=YPAD)
        self.samples_num_lbl_2.pack(side="right", padx=XPAD, pady=YPAD)
        self.samples_lbl_2.pack(side="right", padx=XPAD, pady=YPAD)        
        

class MainApplication(tk.Frame):
    def save_stacks(self, certain_stack, uncertain_stack, image_stack, horz_ratio, vert_ratio):
        out_snippets_certain = []
        out_snippets_uncertain = []

        for page in range(len(certain_stack)):
            for location in certain_stack[page]:
                x = location[0]/vert_ratio
                y = location[1]/horz_ratio
                x1 = int(x - IMAGE_SIZE/2)
                x2 = int(x + IMAGE_SIZE/2)
                y1 = int(y - IMAGE_SIZE/2)
                y2 = int(y + IMAGE_SIZE/2)
                curr_stack = [image_stack[page][chan][y1:y2, x1:x2] for chan in range(len(image_stack[page]))]
                out_snippets_certain.append(curr_stack)    

        for page in range(len(uncertain_stack)):
            for location in uncertain_stack[page]:
                x = location[0]/vert_ratio
                y = location[1]/horz_ratio
                x1 = int(x - IMAGE_SIZE/2)
                x2 = int(x + IMAGE_SIZE/2)
                y1 = int(y - IMAGE_SIZE/2)
                y2 = int(y + IMAGE_SIZE/2)       
                curr_stack = [image_stack[page][chan][y1:y2, x1:x2] for chan in range(len(image_stack[page]))]
                out_snippets_uncertain.append(curr_stack)    
        
        base_name = ntpath.basename(self.in_path.get())
        out_certain = self.out_path.get() + "\\" + base_name + "_certain.pickle"
        out_uncertain = self.out_path.get() + "\\" + base_name + "_uncertain.pickle"

        if out_snippets_certain:
            pickle.dump(out_snippets_certain, open(out_certain, "wb" ))
            print("saved certain snippets to: " + out_uncertain)
        if out_snippets_uncertain:
            pickle.dump(out_snippets_uncertain, open(out_uncertain, "wb" ))
            print("saved uncertain snippets to: " + out_certain)
        
    def add_to_stack(self, stack):
        x, y = self.display.can.x, self.display.can.y
        vert_ratio = self.display.vert_ratio.get()
        horz_ratio = self.display.horz_ratio.get()
        x = min(max(IMAGE_SIZE*vert_ratio/2 + 1, x), self.display.can.winfo_width() - IMAGE_SIZE*vert_ratio/2 - 1)
        y = min(max(IMAGE_SIZE*horz_ratio/2 + 1, y), self.display.can.winfo_height() - IMAGE_SIZE*horz_ratio/2 - 1)
        page = int(self.curr_page.get())
        stack[page].append((x,y))
        self.display.draw_markers(self.certain_stack[page], self.uncertain_stack[page])

        count_certain = sum([len(listElem) for listElem in self.certain_stack])
        count_uncertain = sum([len(listElem) for listElem in self.uncertain_stack])
        self.num_samples_certain.set(str(count_certain))
        self.num_samples_uncertain.set(str(count_uncertain))
        self.display.can.focus_set()

    def remove_from_stack(self):
        x, y = self.display.can.x, self.display.can.y
        def in_dist(ele, x, y):
            x_dif = abs(x - ele[0])
            y_dif = abs(y - ele[1])
            horz_mu = IMAGE_SIZE * self.display.horz_ratio.get() / 2
            vert_mu = IMAGE_SIZE * self.display.vert_ratio.get() / 2
            if y_dif <= horz_mu and x_dif <= vert_mu:
                return True
            return False
        
        page = int(self.curr_page.get())
        self.certain_stack[page] = [ele for ele in self.certain_stack[page] if not in_dist(ele,x,y)]
        self.uncertain_stack[page] = [ele for ele in self.uncertain_stack[page] if not in_dist(ele,x,y)]
        self.display.draw_markers(self.certain_stack[page], self.uncertain_stack[page])
        
        count_certain = sum([len(listElem) for listElem in self.certain_stack])
        count_uncertain = sum([len(listElem) for listElem in self.uncertain_stack])
        self.num_samples_certain.set(str(count_certain))
        self.num_samples_uncertain.set(str(count_uncertain))
        self.display.can.focus_set()

    def update_num_samples(self, num):
        self.num_samples_certain.set(str(num))

    def change_page(self, page):
        self.curr_page.set(str(page))

    def change_layer(self, layer):
        self.curr_layer.set(str(layer))

    def refresh_image(self, page, layer):
        self.curr_layer.set(str(layer))
        self.display.render_image(self.image_stack[page][layer])
        self.display.draw_markers(self.certain_stack[page], self.uncertain_stack[page])
        
    def initialize_image(self, path):
        self.certain_stack.clear()
        self.uncertain_stack.clear()
        self.display.load_img(path, self.image_stack)
        self.curr_page.set("0")
        self.curr_layer.set("0")
        self.num_samples_certain.set("0")
        self.num_samples_uncertain.set("0")
        num_images = len(self.image_stack)
        self.certain_stack = [[] for _ in range(num_images)]
        self.uncertain_stack = [[] for _ in range(num_images)]
        self.display.can.focus_set()
        
    def __init__(self, parent, *args, **kwargs):
        tk.Frame.__init__(self, parent, *args, **kwargs)
        self.in_path = tk.StringVar(self)
        self.out_path = tk.StringVar(self)
        self.curr_layer = tk.StringVar(self)
        self.curr_page = tk.StringVar(self)
        self.num_samples_certain = tk.StringVar(self)
        self.num_samples_uncertain = tk.StringVar(self)


        self.layer = [0, 1, 2, 3]
        self.image_stack = []
        self.certain_stack = []
        self.uncertain_stack = []
        self.image = None
        self.parent = parent

        self.pathbar = PathBar(self)
        self.savebar = SaveBar(self)
        self.layerbar = LayerBar(self)
        self.display = Display(self)
        self.pagebar = PageBar(self)

        self.pathbar.pack(side="top", fill="x")
        self.savebar.pack(side="top", fill="x")
        self.layerbar.pack(side="top", fill="x")
        self.display.pack(side="top", expand=True, fill="both")
        self.pagebar.pack(side="top", fill="x")

        self.in_path.trace_add("write", lambda name, index, mode: self.initialize_image(self.in_path.get()))
        self.curr_layer.trace_add("write", lambda name, index, mode: self.refresh_image(int(self.curr_page.get()), int(self.curr_layer.get())))
        self.curr_page.trace_add("write", lambda name, index, mode: self.refresh_image(int(self.curr_page.get()), int(self.curr_layer.get())))
        self.pagebar.left_btn.bind("<ButtonPress-1>", lambda event: self.change_page(max(0, int(self.curr_page.get()) - 1)))
        self.display.can.bind("q", lambda event: self.change_page(max(0, int(self.curr_page.get()) - 1)))
        self.pagebar.right_btn.bind("<ButtonPress-1>", lambda event: self.change_page(min(len(self.image_stack) - 1, int(self.curr_page.get()) + 1)))
        self.display.can.bind("e", lambda event: self.change_page(min(len(self.image_stack) - 1, int(self.curr_page.get()) + 1)))         
        self.display.can.bind('<ButtonPress-1>', lambda event: self.add_to_stack(self.certain_stack))
        self.display.can.bind('a', lambda event: self.add_to_stack(self.certain_stack))
        self.display.can.bind('<ButtonPress-3>', lambda event: self.add_to_stack(self.uncertain_stack))
        self.display.can.bind('s', lambda event: self.add_to_stack(self.uncertain_stack))
        self.display.can.bind('<ButtonPress-2>', lambda event: self.remove_from_stack())
        self.display.can.bind('d', lambda event: self.remove_from_stack())
        self.savebar.save_btn.bind("<ButtonPress-1>", lambda event: self.save_stacks(self.certain_stack, self.uncertain_stack, self.image_stack, 
                                                                             self.display.horz_ratio.get(), self.display.vert_ratio.get()))
        self.display.can.bind("1", lambda event: self.change_layer("0"))
        self.display.can.bind("2", lambda event: self.change_layer("1"))
        self.display.can.bind("3", lambda event: self.change_layer("2"))
        self.display.can.bind("4", lambda event: self.change_layer("3"))

if __name__ == "__main__":
    root = tk.Tk()
    root.title("Manual Cell Selection Utility")
    root.minsize(1200, 900)
    MainApplication(root).pack(side="top", fill="both", expand=True)
    root.mainloop()