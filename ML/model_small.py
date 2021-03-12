import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Conv2D, concatenate, \
     MaxPool2D, Dropout, Activation
from tensorflow.keras import Model

class MacrophageCNNModel(Model):
    def __init__(self):
        super(MacrophageCNNModel, self).__init__()
        self.conv1 = Conv2D(16, 3, activation='relu')
        self.pool1 = MaxPool2D(pool_size=(2,2))
        self.conv3 = Conv2D(32, 3, activation='relu')
        self.pool2 = MaxPool2D(pool_size=(2,2))
        self.conv4 = Conv2D(64, 3, activation='relu')
        self.pool3 = MaxPool2D(pool_size=(2,2))
        self.flatten = Flatten()
        self.d1 = Dense(64, activation='relu')
        self.drop1 = Dropout(0.2)
        self.d2 = Dense(64,   activation='relu')
        self.drop2 = Dropout(0.2)
        self.d3 = Dense(64, activation='relu')
        self.drop2 = Dropout(0.2)
        self.dout = Dense(3, activation='softmax')

    def call(self, x):
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv3(x)
        x = self.pool2(x)
        x = self.conv4(x)
        x = self.pool3(x)        
        x = self.flatten(x)
        x = self.d1(x)
        x = self.drop1(x)
        x = self.d2(x)
        x = self.drop2(x)
        x = self.d3(x)
        return self.dout(x)

# Create an instance of the model
model = MacrophageCNNModel()
