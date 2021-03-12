import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Conv2D, concatenate, \
     MaxPool2D, Dropout, Activation, BatchNormalization
from tensorflow.keras import Model

class MacrophageCNNModel(Model):
    def __init__(self):
        super(MacrophageCNNModel, self).__init__()
        self.conv1 = Conv2D(64, 3, activation='relu')
        self.pool1 = MaxPool2D(pool_size=(2,2))
        self.bn1 = BatchNormalization()
        self.conv2 = Conv2D(128, 3, activation='relu')
        self.pool2 = MaxPool2D(pool_size=(2,2))
        self.bn2 = BatchNormalization()
        self.conv3 = Conv2D(256, 3, activation='relu')
        self.pool3 = MaxPool2D(pool_size=(2,2))
        self.bn3 = BatchNormalization()
        self.flatten = Flatten()
        self.d1 = Dense(256, activation='relu')
        self.drop1 = Dropout(0.2)
        self.bn4 = BatchNormalization()
        self.d2 = Dense(256,   activation='relu')
        self.drop2 = Dropout(0.2)
        self.bn5 = BatchNormalization()
        self.d3 = Dense(256, activation='relu')
        self.dout = Dense(1, activation='sigmoid')

    def call(self, x):
        x = self.conv1(x)
        x = self.pool1(x)
        #x = self.bn1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        #x = self.bn2(x)
        x = self.conv3(x)
        x = self.pool3(x)        
        #x = self.bn3(x)
        x = self.flatten(x)
        x = self.d1(x)
        x = self.drop1(x)
        x = self.bn4(x)
        x = self.d2(x)
        x = self.drop2(x)
        #x = self.bn5(x)
        x = self.d3(x)
        return self.dout(x)

# Create an instance of the model
model = MacrophageCNNModel()
