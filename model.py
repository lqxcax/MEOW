import keras
from keras import  layers, callbacks, activations, optimizers, losses
from keras.layers import Input, Dense, Dropout, PReLU,\
      Conv2D, SpatialDropout2D, MaxPool2D, BatchNormalization, AveragePooling2D, \
      Flatten, GlobalAveragePooling2D, Concatenate, Add, UpSampling2D, Conv2DTranspose



def model(X_train):
    input  = Input(shape=X_train.shape[1:])

    c1     = Conv2D(32, (3, 3), activation='relu', padding='same')(input)
    bn1    = BatchNormalization()(c1)
    max1   = MaxPool2D((2, 2))(bn1)

    c2     = Conv2D(64, (3, 3), activation='relu', padding='same')(max1)
    bn2    = BatchNormalization()(c2)
    max2   = MaxPool2D((2, 2))(bn2)

    c3     = Conv2D(128, (3, 3), activation='relu', padding='same')(max2)

    ct1    = Conv2DTranspose(64, (3, 3), strides=(2, 2), activation='relu', padding='same')(c3)
    concat = Concatenate()([ct1, bn2])

    ct2    = Conv2DTranspose(32, (3, 3), strides=(2, 2), activation='relu', padding='same')(concat)
    concat = Concatenate()([ct2, bn1])

    output = Conv2D(3, 1, activation='sigmoid', padding='same')(concat)


    model_color = keras.Model(inputs=input, outputs=output)
    
    return model_color