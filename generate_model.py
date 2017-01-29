from keras.models import Sequential
from keras.layers import Convolution2D, Flatten, Dense, Dropout
from keras.regularizers import l2
from keras.constraints import maxnorm
from bc_utils import *

def generate_model():
    model = Sequential()
    model.add(Convolution2D(24, 5, 5, subsample=(2, 2), input_shape=IMG_SHAPE))
    model.add(Convolution2D(36, 5, 5, subsample=(2, 2)))
    model.add(Convolution2D(48, 5, 5, subsample=(2, 2)))
    model.add(Convolution2D(64, 3, 3))
    model.add(Convolution2D(64, 3, 3))
    model.add(Flatten())
    model.add(Dense(100, W_regularizer=l2(0.01), W_constraint=maxnorm(4), activation='relu'))
    model.add(Dense(50, W_regularizer=l2(0.01), W_constraint=maxnorm(4), activation='relu'))
    model.add(Dense(10, W_regularizer=l2(0.01), W_constraint=maxnorm(4)))
    model.add(Dense(1, W_regularizer=l2(0.01), W_constraint=maxnorm(4)))

    model.compile(optimizer='adam',
                  loss='mse',
                  metrics=['mean_squared_error'])
    return model

if __name__ == '__main__':
    model = generate_model()
    with open('model.json', 'w') as f:
        f.write(model.to_json())
