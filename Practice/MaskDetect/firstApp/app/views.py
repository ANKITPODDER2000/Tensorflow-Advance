from django.shortcuts import render , HttpResponse
from django.core.files.storage import FileSystemStorage

import tensorflow as tf
import matplotlib.pyplot as plt
from time import sleep
import cv2
import numpy as np

face_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
model = tf.keras.models.load_model("mask_detect.h5")
classes = {0 : 'WithMask', 1 : 'WithoutMask'}
# Create your views here.

def index(request):
    context={'a':1}
    return render(request,'index.html',context)

def predictImage(request):
    #print (request)
    #print (request.POST.dict())
    fileObj=request.FILES['filePath']
    fs=FileSystemStorage()
    filePathName=fs.save(fileObj.name,fileObj)
    filePathName=fs.url(filePathName)
    testimage='.'+filePathName

    img = cv2.imread(testimage)
    img = cv2.resize(img , (450 , 450 ))
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray)
    print(faces)
    
    for (x,y,w,h) in faces:
        cv2.rectangle(gray,(x,y),(x+w,y+h),(255,0,0),2)
        roi_gray = None
        try:
            roi_gray = img[y-60:y+h+15,x:x+w]
        except:
            try:
                roi_gray = img[y-50:y+h,x:x+w]
            except:
                try:
                    roi_gray = img[y-40:y+h,x:x+w]
                except:
                    try:
                        roi_gray = img[y-25:y+h,x:x+w]
                    except:
                        try:
                            roi_gray = img[y-15:y+h,x:x+w]
                        except:
                            try:
                                roi_gray = img[y-5:y+h,x:x+w]
                            except:
                                try:
                                    roi_gray = img[y:y+h,x:x+w]
                                except:
                                    pass
                                
        roi_gray = cv2.resize(roi_gray,(150,150))
        if np.sum([roi_gray])!=0:
            roi = roi_gray.astype('float')/255.0
            roi = tf.keras.preprocessing.image.img_to_array(roi)
            roi = np.expand_dims(roi,axis=0)

            preds = model.predict_classes(roi)[0][0]
            print(preds)
        else:
            pass

    

            
    return render(request , "predict.html",{'context' : True})


    
