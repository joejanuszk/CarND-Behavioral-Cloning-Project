import os
import gc
import time
import cv2
import numpy as np
import matplotlib.image as mpimg
from generate_model import generate_model
from bc_utils import *

PATH_TO_CSV = '../data/driving_log.csv'
IMG_BASE = '../data/'
INPUT_LAYER = 'convolution2d_input_1'
OUTPUT_LAYER = 'dense_4'

FLIP_ABOUT_Y_AXIS = 1

ZERO_IMAGES_RATIO = 0.05
ANGLE_ADJUST = 0.28
ZERO_DELTA = 0.1
NOISE_STDDEV = 0.04

def is_near_zero(val):
    return abs(val['angle']) < ZERO_DELTA

def is_not_near_zero(val):
    return not is_near_zero(val)

def add_noise_to_angle(angle):
    return angle + np.random.normal(0, NOISE_STDDEV)

def get_better_training_data(lines):
    data = []
    for line in lines:
        parts = line.split(', ')
        center_img_path = parts[0]
        left_img_path = parts[1]
        right_img_path = parts[2]
        angle = float(parts[3])# + np.random.normal(0, 0.1)
        data.append({'path': center_img_path, 'angle': add_noise_to_angle(angle), 'reverse': False})
        data.append({'path': center_img_path, 'angle': add_noise_to_angle(-angle), 'reverse': True})
        l_angle_adj = angle + ANGLE_ADJUST
        r_angle_adj = angle - ANGLE_ADJUST
        if (l_angle_adj > 1.):
            l_angle_adj = 1.
        if (r_angle_adj < -1.):
            r_angle_adj = -1.
        data.append({'path': left_img_path, 'angle': add_noise_to_angle(l_angle_adj), 'reverse': False})
        data.append({'path': left_img_path, 'angle': add_noise_to_angle(-l_angle_adj), 'reverse': True})
        data.append({'path': right_img_path, 'angle': add_noise_to_angle(r_angle_adj), 'reverse': False})
        data.append({'path': right_img_path, 'angle': add_noise_to_angle(-r_angle_adj), 'reverse': True})
        #data.append({'path': center_img_path, 'angle': angle, 'reverse': False})
        #data.append({'path': center_img_path, 'angle': -angle, 'reverse': True})
        #l_angle_adj = angle + ANGLE_ADJUST
        #r_angle_adj = angle - ANGLE_ADJUST
        #if (l_angle_adj > 1.):
        #    l_angle_adj = 1.
        #if (r_angle_adj < -1.):
        #    r_angle_adj = -1.
        ##if (l_angle_adj <= 1.):
        #data.append({'path': left_img_path, 'angle': l_angle_adj, 'reverse': False})
        #data.append({'path': left_img_path, 'angle': -l_angle_adj, 'reverse': True})
        ##if (r_angle_adj >= -1.):
        #data.append({'path': right_img_path, 'angle': r_angle_adj, 'reverse': False})
        #data.append({'path': right_img_path, 'angle': -r_angle_adj, 'reverse': True})
    nonzeros = list(filter(is_not_near_zero, data))
    zeros = list(filter(is_near_zero, data))
    np.random.shuffle(zeros)
    some_zeros = zeros[:int(len(nonzeros) * ZERO_IMAGES_RATIO)]
    all_data = nonzeros + some_zeros
    np.random.shuffle(all_data)
    return all_data

def generate_arrays_from_file(path):
    while True:
        with open(PATH_TO_CSV, 'r') as f:
            lines = list(map(lambda l: l, f))
        data = get_better_training_data(lines)
        np.random.shuffle(data)
        for entry in data:
            processed_img = load_and_process_image(entry['path'],
                                                   entry['reverse'],
                                                   filename_only=True)
            x = np.reshape(processed_img, (1, TOTAL_Y, RESIZED_X, COLORS))
            y = np.reshape(entry['angle'], (1, 1))
            # TODO name the layers something non-automatic
            yield ({INPUT_LAYER: x}, {OUTPUT_LAYER: y})

def normalize_image_features(im):
    return (im / 255.) - 0.5

def process_image(im):
    resized_image = cv2.resize(im, dsize=(RESIZED_X, RESIZED_Y))
    cropped_image = resized_image[VTRIMTOP:RESIZED_Y-VTRIMBOT, :]
    yuv_image = cv2.cvtColor(cropped_image, cv2.COLOR_RGB2YUV)
    return normalize_image_features(yuv_image)
    #return normalize_image_features(cropped_image)
    #gray_image = cv2.cvtColor(cropped_image, cv2.COLOR_RGB2GRAY)
    #return normalize_image_features(gray_image)

def load_and_process_image(filename, reverse, filename_only=False):
    if (filename_only):
        full_path = os.path.join(IMG_BASE, filename)
    else:
        full_path = filename
    raw_image = mpimg.imread(full_path)
    if (reverse):
        raw_image = cv2.flip(raw_image, FLIP_ABOUT_Y_AXIS)
    return process_image(raw_image)

if __name__ == '__main__':
    model = generate_model()

    start_time = time.time()
    model.fit_generator(
        generate_arrays_from_file(PATH_TO_CSV),
        samples_per_epoch=3000,
        nb_epoch=5,
        verbose=1,
        nb_val_samples=500,
        #max_q_size=1, # fix issue where multiple generators break stuff on non-GPU machine
        validation_data=generate_arrays_from_file(PATH_TO_CSV))
    print("--- %s seconds ---" % (time.time() - start_time))

    model.save_weights('model.h5')

    score = model.evaluate_generator(
        generate_arrays_from_file(PATH_TO_CSV),
        500)
    print('Test score:', score[0])
    print('Test accuracy:', score[1])
    #gc.collect()
