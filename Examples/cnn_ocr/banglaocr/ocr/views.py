from django.shortcuts import render,render_to_response
from keras.models import load_model
from django.http import HttpResponse

import base64
import cv2
import numpy as np
# import keras
model=None
labels=[9,8,2,3,6,1,7,5,4,0]
def b64_2_im(uri):
    data = uri.split(',')[1]
    data=base64.decodebytes(bytes(data, "utf-8"))
    g = open("temp.jpg", "wb")
    g.write(data)
    g.close()
    im=cv2.imread('temp.jpg',0)
    im=cv2.resize(im,(30,30))
    im=np.reshape(im,(im.shape[0],im.shape[1],1))
    im=im/255
    return im


def index(request):
    return render_to_response('index.html')

def predict(request):
    data=request.POST.get('im')
    im=b64_2_im(data)
    global model
    if model is None:
        model= load_model('model.h5')
    res=np.argmax(model.predict(np.array([im])))
    return HttpResponse("Prediction : "+str(res));
