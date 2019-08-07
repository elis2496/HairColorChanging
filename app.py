import sys
import os
import glob
import re
import numpy as np
from model_nn.colours import colors
import io
import cv2
import base64
import json
from io import BytesIO

from keras.applications.imagenet_utils import preprocess_input, decode_predictions
from keras.models import load_model
from keras.preprocessing import image
from PIL import Image


from flask import Flask, redirect, url_for, request, render_template, jsonify
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer

from model_nn.inference import Inference
from model_nn.unet_model import VGGUnet

INPUT_SHAPE = (224, 224, 3)


model = VGGUnet(INPUT_SHAPE, mode='inference')
CUR_DIR = os.path.abspath('./')
WEIGHTS_PATH = os.path.join(CUR_DIR, 'model_nn', 'weights.h5py')

model.load_weights(WEIGHTS_PATH)
color = colors['red1']

app = Flask(__name__)


def model_predict(img):
    inference = Inference(model, color)
    preds = inference.predict(img)
    res_image = Image.fromarray(np.uint8(preds))
    buffered = io.BytesIO()
    res_image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue())
    return img_str

@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        f = request.get_json()
        basepath = os.path.dirname(__file__)
        b_64_im = f['file']
        encoded_image = b_64_im.split(",")[1]
        decoded_image = base64.b64decode(encoded_image)
        img = Image.open(BytesIO(decoded_image))
        img = img.resize((224, 224), Image.ANTIALIAS)
        img = np.array(img)[:,:,:3]
        preds = model_predict(img)
        response = {}
        response['file'] = "data:image/png;base64," + preds.decode()
        return json.dumps(response)
    return None


if __name__ == '__main__':
    # app.run(port=5002, debug=True)

    # Serve the app with gevent
    http_server = WSGIServer(('', 5000), app)
    http_server.serve_forever()
