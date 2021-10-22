from PIL import Image, ImageTk, ImageFile
import numpy as np
from tkinter import (Tk, Frame, Button, Label, StringVar, LEFT, TOP, BOTH,
                     BOTTOM)
from tkinter import filedialog
from glob import glob
from functools import reduce
from operator import add
import pickle
import re
from collections import defaultdict

ImageFile.LOAD_TRUNCATED_IMAGES = True


class Annotator:
    def load_image(self):
        if self.img_id_list:
            self.img_id = self.img_id_list[self.index]
            self.title = self.img_dict[self.img_id][0]
            self.img = np.asarray(Image.open(self.title))
            self.tk_img = self.np2img(self.img)
            self.la.config(image=self.tk_img, bg="#000000",
                           width=1024, height=1024)
            self.tv.set(self.title)

    def np2img(self, data, size=(1024, 1024)):
        im = Image.fromarray(data, 'L' if data.ndim == 2 else 'RGB')
        if size:
            im = im.resize(size)
        return ImageTk.PhotoImage(image=im)

    def select_folder(self):
        folder = filedialog.askdirectory()
        if folder != "":
            img_ext = ['jpg', 'tif', 'png', 'jpeg']
            img_list = reduce(add,
                              [sorted(glob(folder+'/*.{}'.format(ext)))
                               for ext in img_ext])
            for im in img_list:
                self.img_dict[re.match('.*?(\d+)', im).group(1)].append(im)
            self.img_id_list = list(self.img_dict.keys())
        self.load_image()

    def _prev(self):
        self.index = max(0, self.index - 1)
        self.load_image()

    def _next(self):
        self.index = min(len(self.img_id_list)-1, self.index+1)
        self.load_image()

    def _class0(self):
        self.classification_dict[self.title] = 0
        if len(self.img_dict[self.img_id]) > 1:
            title = self.img_dict[self.img_id][1]
            self.classification_dict[title] = 1
            print(title, 1)
        print(self.title, 0,
              'num_labeled: {}'.format(len(self.classification_dict)))

    def _class1(self):
        self.classification_dict[self.title] = 1
        if len(self.img_dict[self.img_id]) > 1:
            title = self.img_dict[self.img_id][1]
            self.classification_dict[title] = 0
            print(title, 0)
        print(self.title, 1,
              'num_labeled: {}'.format(len(self.classification_dict)))

    def _save(self):
        pickle.dump(self.classification_dict, open('labels.pkl', 'wb'))
        print("labels saved to labels.pkl")

    def __init__(self):
        self.classification_dict = {}
        self.img_dict = defaultdict(list)
        self.index = 0
        self.root = Tk()
        self.root.resizable(False, False)
        self.root.title('Binary Classifier')
        self.root.bind('<Left>', lambda x: self._prev())
        self.root.bind('<Right>', lambda x: self._next())
        self.root.bind('<z>', lambda x: self._class0())
        self.root.bind('<x>', lambda x: self._class1())
        self.root.bind('<Return>', lambda x: self._save())
        self.tv = StringVar()
        frame = Frame(self.root)
        Button(frame, text="Open Folder",
               command=self.select_folder).pack(side=LEFT)
        Button(frame, text="Prev", command=self._prev).pack(side=LEFT)
        Button(frame, text="Next", command=self._next).pack(side=LEFT)
        Label(frame, textvariable=self.tv).pack(side=LEFT)
        frame.pack(side=TOP, fill=BOTH)
        self.la = Label(self.root, width=100, height=100)
        self.la.pack(side=LEFT)
        right_frame = Frame(self.root).pack()
        Button(right_frame, text="class_0",
               command=self._class0).pack()
        Button(right_frame, text="class_1",
               command=self._class1).pack()
        Button(right_frame, text="save", command=self._save).pack(side=BOTTOM)


if __name__ == "__main__":
    s = Annotator()
    s.root.mainloop()
