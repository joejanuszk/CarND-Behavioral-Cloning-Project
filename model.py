import os
import gc
import cv2
import numpy as np
import matplotlib.image as mpimg
from keras.models import Sequential
from keras.layers import Convolution2D, Flatten, Dense

PATH_TO_CSV = '../driving_log.csv'
IMG_BASE = '../IMG/'
RESIZED_X = 160
RESIZED_Y = 80
IMG_SHAPE = (RESIZED_Y, RESIZED_X, 3)

def generate_arrays_from_file(path):
    while True:
        with open(PATH_TO_CSV, 'r') as f:
            for line in f:
                parts = line.split(', ')
                center_img_path = parts[0]
                angle = parts[3]
                processed_img = load_and_normalize_image(center_img_path)
                x = np.reshape(processed_img, (1, RESIZED_Y, RESIZED_X, 3))
                y = np.reshape(angle, (1, 1))
                # TODO name the layers something non-automatic
                yield ({'convolution2d_input_1': x}, {'dense_4': y})

def normalize_image_features(im):
    return (im / 255.) - 0.5

def load_and_normalize_image(filename, filename_only=False):
    if (filename_only):
        full_path = os.path.join(IMG_BASE, filename)
    else:
        full_path = filename
    raw_image = mpimg.imread(full_path)
    resized_image = cv2.resize(raw_image, dsize=(RESIZED_X, RESIZED_Y))
    return normalize_image_features(resized_image)

model = Sequential()
model.add(Convolution2D(24, 5, 5, subsample=(2, 2), input_shape=IMG_SHAPE))
model.add(Convolution2D(36, 5, 5, subsample=(2, 2)))
model.add(Convolution2D(48, 5, 5, subsample=(2, 2)))
model.add(Convolution2D(64, 3, 3))
model.add(Convolution2D(64, 3, 3))
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))

model.compile(optimizer='rmsprop',
              loss='mse',
              metrics=['accuracy'])

model.fit_generator(
    generate_arrays_from_file(PATH_TO_CSV),
    samples_per_epoch=20,
    nb_epoch=5,
    #batch_size=32,
    verbose=1,
    nb_val_samples=20,
    max_q_size=1, # fix issue where multiple generators break stuff
    validation_data=generate_arrays_from_file(PATH_TO_CSV))

score = model.evaluate_generator(
    generate_arrays_from_file(PATH_TO_CSV),
    20)
print('Test score:', score[0])
print('Test accuracy:', score[1])
gc.collect()
