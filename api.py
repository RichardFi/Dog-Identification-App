from flask import Flask, request, jsonify, flash
from keras.models import load_model
import numpy as np
import pandas as pd
import traceback
import cv2
from keras import Model
from keras.applications import InceptionV3, Xception
from keras.layers import Input, GlobalAveragePooling2D, Lambda
from keras.applications.inception_v3 import preprocess_input
from werkzeug.utils import secure_filename
import os

app = Flask(__name__)

UPLOAD_FOLDER = "./images/"
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route("/predict", methods=['POST'])
def predict():
    if pred_model:
        try:
            #json_ = request.json
            myimg = request.files['file']

            print(myimg)

            filename = secure_filename(myimg.filename) # save file 
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            myimg.save(filepath)

            def get_features(MODEL, data):
                # weights='imagenet'
                cnn_model = MODEL(include_top=False, input_shape=(width, width, 3), weights='imagenet')
                
                inputs = Input((width, width, 3))
                x = inputs
                # preprocess_input to modify input format for the model
                x = Lambda(preprocess_input, name='preprocessing')(x)
                x = cnn_model(x)
                x = GlobalAveragePooling2D()(x)
                cnn_model = Model(inputs, x)

                features = cnn_model.predict(data, batch_size=64, verbose=1)
                return features

            width = 299
            # the number of images need to be predict
            numImage = 1
            X_predict = np.zeros((numImage, width, width, 3), dtype=np.uint8)

            # read the image
            pred_img = "images/" + filename
            X_predict[0] = cv2.resize(cv2.imread(pred_img), (width, width))

            inception_features_predict = get_features(InceptionV3, X_predict)
            xception_features_predict = get_features(Xception, X_predict)
            features_predict = np.concatenate([inception_features_predict, xception_features_predict], axis=-1)

            y_predict = pred_model.predict(features_predict, batch_size=128) 
            #result_prob = np.max(y_predict[0])
            result = np.argmax(y_predict[0])

            return jsonify({'prediction': breed[result]})
        except:
            return jsonify({'trace': traceback.format_exc()})
    else:
        print ('Train the model first')
        return ('No model here to use')

if __name__ == '__main__':
    try:
        port = int(sys.argv[1]) # This is for a command-line input
    except:
        port = 5000 # If you don't provide any port the port will be set to 12345

    pred_model = load_model("model") # Load "model.pkl"
    print ('Model loaded')

    df_breed = pd.read_csv('data/breed.csv')
    breed = df_breed['0'].tolist()

    # number of dog breeds
    #n_class = len(breed)

    #class_to_num = dict(zip(breed, range(n_class)))
    #num_to_class = dict(zip(range(n_class), breed))

    app.run(port=port, debug=True)