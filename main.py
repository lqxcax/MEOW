import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_absolute_error

import tensorflow as tf
import keras
from keras import  layers, callbacks, activations, optimizers, losses
from keras.layers import Input, Dense, Dropout, PReLU,\
      Conv2D, SpatialDropout2D, MaxPool2D, BatchNormalization, AveragePooling2D, \
      Flatten, GlobalAveragePooling2D, Concatenate, Add, UpSampling2D, Conv2DTranspose

from load_dataset import load_data
from model import model

gray, color = load_data('landscape Images', 128)

gray  = np.array(gray)
color = np.array(color)

X_train, X_test, y_train, y_test = train_test_split(gray, color, test_size=0.2, random_state=1)

X_train = X_train[..., np.newaxis]
X_test  = X_test[..., np.newaxis]

model_color = model(X_train)

adam  = optimizers.Adam(learning_rate=0.001)
huber = losses.Huber()

model_color.compile(
    optimizer=adam,
    loss=huber,
)

es = callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
rlr = callbacks.ReduceLROnPlateau(monitor='val_loss', patience=3, factor=0.1)

model_color.fit(
    X_train, y_train, 
    
    validation_split = 0.2,
    epochs = 50,
    callbacks=[es, rlr]
)


y_pred = model_color.predict(X_test)
print(mean_absolute_error(y_test, y_pred))

