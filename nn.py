import tensorflow as tf
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D, Flatten
from tensorflow.keras import Sequential
from tensorflow.keras.optimizers import RMSprop
from keras.losses import categorical_crossentropy

class NN:
    def __init__(self, action_space):
        self.output_shape = action_space.n
        self.model = Sequential([
            Conv2D(8, 3, activation='relu', data_format='channels_last', input_shape=(210,160,1)),
            MaxPool2D(2),
            Conv2D(16,3, activation='relu'),
            MaxPool2D(2),
            Conv2D(32,3, activation='relu'),
            MaxPool2D(2),
            Conv2D(64,3, activation='relu'),
            MaxPool2D(2),
            Conv2D(64,3, activation='relu'),
            MaxPool2D(2),
            Flatten(),
            Dense(64),
            Dense(self.output_shape)
        ])
        self.model.summary()
        exit()
        self.model.compile(loss='categorical_crossentropy', optimizer='adam')
