from keras.applications import inception_v3,imagenet_utils
import cv2, numpy as np
from flask import Flask, request, make_response,jsonify
import numpy as np
import json
import urllib

model = None
app = Flask(__name__,static_url_path='')

def preprocess_img(im,target_size=(299,299)):
    img=cv2.resize(im,target_size)
    img=np.divide(img,255.)
    img=np.subtract(img,0.5)
    img=np.multiply(img,2.)
    return img

def load_im_from_url(url):
    requested_url = urllib.urlopen(url)
    image_array = np.asarray(bytearray(requested_url.read()), dtype=np.uint8)
    img = cv2.imdecode(image_array, -1)
    return img

def predict(url):
    img=load_im_from_url(url)
    img=preprocess_img(img)
    global model
    if model is None:
        model =inception_v3.InceptionV3()
        model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
    preds = model.predict(np.array([img]))
    return imagenet_utils.decode_predictions(preds)

@app.route('/classify', methods=['GET'])
def classify():
    image_url = request.args.get('imageurl')
    resp=predict(image_url)
    result=[]
    for r in resp[0]:
        result.append({"class_name":r[1],"score":float(r[2])})
    return jsonify({'results':result})

@app.route('/', methods=['GET'])
def root():
    return app.send_static_file('index.html')
if __name__ == '__main__':
    app.run(debug=True)
