import cv2
import numpy as np
from tqdm import tqdm
import pandas as pd

import random
import matplotlib.pyplot as plt

from keras.layers import *
from keras.models import *
from keras.applications import *
from keras.optimizers import *
from keras.regularizers import *
from keras.applications.inception_v3 import preprocess_input

#from IPython.display import SVG
#from keras.utils.vis_utils import model_to_dot
#import pydot_ng as pydot

import matplotlib.pyplot as plt

# Data preprocessing
df = pd.read_csv('data/labels.csv')

# number of images in train set
n = len(df)
# all names of dog breeds
breed = set(df['breed'])
# number of dog breeds
n_class = len(breed)

class_to_num = dict(zip(breed, range(n_class)))
num_to_class = dict(zip(range(n_class), breed))

# Images need to be resized to a same size for trainning
# The width of image after resizing, 299 is the default input size for the models.
width = 299

# to store images after cv2 transfer
X = np.zeros((n, width, width, 3), dtype=np.uint8)
y = np.zeros((n, n_class), dtype=np.uint8)

# tqdm is used to display the process
for i in tqdm(range(n)):
    # '%'
    X[i] = cv2.resize(cv2.imread('data/train/%s.jpg' % df['id'][i]), (width, width))
    y[i][class_to_num[df['breed'][i]]] = 1


'''
# visualize sample images
plt.figure(figsize=(12, 6))
for i in range(10):
    random_index = random.randint(0, n-1)
    plt.subplot(2, 5, i+1)
    # BGR to RGB
    plt.imshow(X[random_index][:,:,::-1])
    plt.title(num_to_class[y[random_index].argmax()])
'''

# get features from InceptionV3 and Xception
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

inception_features = get_features(InceptionV3, X)
xception_features = get_features(Xception, X)
features = np.concatenate([inception_features, xception_features], axis=-1)

# Train model
inputs = Input(features.shape[1:])
x = inputs
x = Dropout(0.5)(x)
x = Dense(n_class, activation='softmax')(x)
model = Model(inputs, x)
model.compile(optimizer='nadam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
h = model.fit(features, y, batch_size=128, epochs=10, validation_split=0.1)
