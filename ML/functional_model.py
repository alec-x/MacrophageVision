import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Conv2D, concatenate, \
     MaxPool2D, Dropout, Activation
from tensorflow.keras import Model
def get_model(channels):
    img_inputs = keras.Input(shape=(32, 32, channels))
    conv1 = Conv2D(64, kernel_size=3, activation='relu')(img_inputs)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = Conv2D(128, kernel_size=3, activation='relu')(pool1)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = Conv2D(256, kernel_size=3, activation='relu')(pool2)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    flat = Flatten()(pool3)
    hidden1 = Dense(256, activation='relu')(flat)
    drop1 = Dropout(0.2)(hidden1)
    hidden2 = Dense(256, activation='relu')(flat)
    drop2 = Dropout(0.2)(hidden1)
    hidden3 = Dense(256, activation='relu')(flat)
    output = Dense(1, activation='sigmoid')(hidden3)
    model = Model(inputs=img_inputs, outputs=output)
    
    return model

# summarize layers
