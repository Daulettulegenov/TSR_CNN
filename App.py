# -*- coding: utf-8 -*-
"""
Created on Tue Mar 29 15:24:01 2022

@author: Tulegenov Daulet
"""

import tkinter as tk
from tkinter import filedialog
from tkinter import *
from PIL import ImageTk, Image
import cv2
import numpy as np
from keras.models import load_model

#load the trained model to classify sign
model = load_model('CNN_model_3.h5')
#model = load_model('my_model_t.h5')

classes = { 1:'Speed limit (20km/h)',
            2:'Speed limit (30km/h)',      
            3:'Speed limit (50km/h)',       
            4:'Speed limit (60km/h)',      
            5:'Speed limit (70km/h)',    
            6:'Speed limit (80km/h)',      
            7:'End of speed limit (80km/h)',     
            8:'Speed limit (100km/h)',    
            9:'Speed limit (120km/h)',     
           10:'No passing',   
           11:'No passing veh over 3.5 tons',     
           12:'Right-of-way at intersection',     
           13:'Priority road',    
           14:'Yield',     
           15:'Stop',       
           16:'No vehicles',       
           17:'Veh > 3.5 tons prohibited',       
           18:'No entry',       
           19:'General caution',     
           20:'Dangerous curve left',      
           21:'Dangerous curve right',   
           22:'Double curve',      
           23:'Bumpy road',     
           24:'Slippery road',       
           25:'Road narrows on the right',  
           26:'Road work',    
           27:'Traffic signals',      
           28:'Pedestrians',     
           29:'Children crossing',     
           30:'Bicycles crossing',       
           31:'Beware of ice/snow',
           32:'Wild animals crossing',      
           33:'End speed + passing limits',      
           34:'Turn right ahead',     
           35:'Turn left ahead',       
           36:'Ahead only',      
           37:'Go straight or right',      
           38:'Go straight or left',      
           39:'Keep right',     
           40:'Keep left',      
           41:'Roundabout mandatory',     
           42:'End of no passing',      
           43:'End no passing veh > 3.5 tons' }

#initialise GUI
w=tk.Tk()
w.geometry('500x200')
w.title('Traffic sign recognition')
w.configure(background='#99a6bf')

def byimg():
    w.destroy()
    tsr1=tk.Tk()
    tsr1.geometry('800x500')
    tsr1.title('Traffic sign classification by image')
    tsr1.configure(background='#99a6bf')
    
    label=Label(tsr1,background='#99a6bf', font=('arial',20,'italic'))
    sign_image = Label(tsr1)
    
    #PREPROCESSING THE IMAGES
 
    def grayscale(img):
        img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        return img
    
    def equalize(img):
        img =cv2.equalizeHist(img)
        return img
    
    def preprocessing(img):
        img = grayscale(img)     # CONVERT TO GRAYSCALE
        img = equalize(img)      # STANDARDIZE THE LIGHTING IN AN IMAGE
        img = img/255            # TO NORMALIZE VALUES BETWEEN 0 AND 1 INSTEAD OF 0 TO 255
        return img

    def classify(file_path):
        global label_packed
        image = cv2.imread(file_path)
        image = np.asarray(image)
        image = cv2.resize(image, (30, 30))
        image = preprocessing(image)
        image = image.reshape(1, 30, 30, 1)
        print(image.shape)
        predict_x=model.predict(image)
        pred=np.argmax(predict_x)
        sign = classes[pred+1]
        print(sign)
        label.configure(foreground='#011638', text='Class: ' + sign)
        
        

    def upload_image():
        global file_path
        try:
            file_path=filedialog.askopenfilename()
            uploaded=Image.open(file_path)
            uploaded.thumbnail(((tsr1.winfo_width()/2.25),(tsr1.winfo_height()/2.25)))
            im=ImageTk.PhotoImage(uploaded)
            sign_image.configure(image=im)
            sign_image.image=im
            label.configure(text='')
        except:
            pass
        
    heading = Label(tsr1, text="Traffic sign recognition by image",
                    pady=20, font=('arial',20,'italic', 'bold'))
    heading.configure(background='#99a6bf',foreground='#000000')
    heading.pack()
    upload=Button(tsr1,text="Upload an image",
                  command=upload_image,padx=10,pady=5)
    upload.configure(background='#C8A2C8', 
                     foreground='#000000',font=('arial',13,'italic'))
    upload.place(relx=.45, rely=.12, anchor="ne", height=30, width=200)
    classify_b=Button(tsr1,text="Classify Image",
                      command=lambda: classify(file_path),padx=10,pady=5)
    classify_b.configure(background='#C8A2C8', 
                         foreground='#000000',font=('arial',13,'italic'))
    classify_b.place(relx=.55, rely=.12, anchor="nw", height=30, width=200)
    sign_image.pack(side=TOP,expand=True)
    label.pack(expand = True)
    tsr1.mainloop()


def bycam():
    w.destroy()
    execfile('tc_2.py')
    
butt1 = Button(w, text='Traffic sign recognition by image',background="#C8A2C8", 
               foreground="#000000", font=('arial',13,'italic'), command=byimg)
butt2 = Button(w, text='Traffic sign recognition in real time',background="#C8A2C8", 
               foreground="#000000", font=('arial',13,'italic'), command=bycam)
butt1.place(y = 50, relx=.5, rely=.05, anchor="c", 
            height=30, width=300, bordermode=OUTSIDE)
butt2.place(y = 50, relx=.5, rely=.4, anchor="c", 
            height=30, width=300, bordermode=OUTSIDE)

w.mainloop()



