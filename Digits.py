#!/usr/bin/env python
# coding: utf-8

# author: Jordan King

from __future__ import absolute_import
from __future__ import print_function

import numpy as np
import pandas as pd
import csv as csv

from sklearn.cross_validation import KFold
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import OneHotEncoder

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras.preprocessing.image import ImageDataGenerator
from keras.layers.advanced_activations import PReLU
from keras.callbacks import ModelCheckpoint
from keras.utils.generic_utils import Progbar, printv

from random import randint, uniform

from matplotlib import pyplot
from skimage.io import imshow
from skimage.util import crop
from skimage import transform, filters, exposure

def load_data(filename, nb_classes, subset = 1):
    """ returns X and y - data and target - as numpy arrays, X normalized
    and y made categorical. """

    data = np.genfromtxt(filename, delimiter=',', skip_header=1, dtype='float32')

    if subset != 1:
        np.random.shuffle(data)
        data = data[0:int(subset*data.shape[0]):, ::]

    X = data[0::,1::]
    X = X.reshape(X.shape[0], 1, 28, 28)
    X /= 255

    y = data[0::,0]
    y = np_utils.to_categorical(y, nb_classes)

    

    datagen = ImageDataGenerator(featurewise_center=False,
                                        samplewise_center=False,
                                        featurewise_std_normalization=False,
                                        samplewise_std_normalization=False,
                                        zca_whitening=False,
                                        rotation_range=0,
                                        width_shift_range=0.,
                                        height_shift_range=0.,
                                        horizontal_flip=False,
                                        vertical_flip=False)
    datagen.fit(X)

    return [X, y, datagen]

#### adapted from https://github.com/FlorianMuellerklein/lasagne_mnist/blob/master/helpers.py

def batch_warp(X_batch, y_batch):
    '''
    Data augmentation for feeding images into CNN.
    This example will randomly rotate all images in a given batch between -10 and 10 degrees
    and to random translations between -5 and 5 pixels in all directions.
    Random zooms between 1 and 1.3.
    Random shearing between -20 and 20 degrees.
    Randomly applies sobel edge detector to 1/4th of the images in each batch.
    Randomly inverts 1/2 of the images in each batch.
    '''
    PIXELS = 28

    # set empty copy to hold augmented images so that we don't overwrite
    X_batch_aug = np.empty(shape = (X_batch.shape[0], 1, PIXELS, PIXELS), dtype = 'float32')

    # random rotations betweein -8 and 8 degrees
    dorotate = randint(-5,5)

    # random translations
    trans_1 = randint(-3,3)
    trans_2 = randint(-3,3)

    # random zooms
    zoom = uniform(0.8, 1.2)

    # shearing
    shear_deg = uniform(-10, 10)

    # set the transform parameters for skimage.transform.warp
    # have to shift to center and then shift back after transformation otherwise
    # rotations will make image go out of frame
    center_shift   = np.array((PIXELS, PIXELS)) / 2. - 0.5
    tform_center   = transform.SimilarityTransform(translation=-center_shift)
    tform_uncenter = transform.SimilarityTransform(translation=center_shift)

    tform_aug = transform.AffineTransform(rotation = np.deg2rad(dorotate),
                                          scale =(1/zoom, 1/zoom),
                                          shear = np.deg2rad(shear_deg),
                                          translation = (trans_1, trans_2))

    tform = tform_center + tform_aug + tform_uncenter

    # images in the batch do the augmentation
    for j in range(X_batch.shape[0]):
        X_batch_aug[j][0] = transform._warps_cy._warp_fast(X_batch[j][0], tform.params, (PIXELS, PIXELS))

        # return transform._warps_cy._warp_fast(X_batch[j][0], tform.params, (PIXELS, PIXELS))


    # use sobel edge detector filter on one quarter of the images
    indices_sobel = np.random.choice(X_batch_aug.shape[0], X_batch_aug.shape[0] / 4, replace = False)
    for k in indices_sobel:
        img = X_batch_aug[k][0]
        X_batch_aug[k][0] = filters.sobel(img)

    return [X_batch_aug, y_batch]

#### end adaption from https://github.com/FlorianMuellerklein/lasagne_mnist/blob/master/helpers.py

def build_keras(nb_classes):

    model = Sequential()

    model.add(Convolution2D(64, 1, 3, 3, border_mode='full')) 
    model.add(Activation('relu'))

    model.add(Convolution2D(64, 64, 3, 3))
    model.add(Activation('relu'))

    model.add(MaxPooling2D(poolsize=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Convolution2D(128, 64, 3, 3, border_mode='full')) 
    model.add(Activation('relu'))

    model.add(Convolution2D(128, 128, 3, 3))
    model.add(Activation('relu'))

    model.add(MaxPooling2D(poolsize=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())

    model.add(Dense(32*196, 512))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    model.add(Dense(512, 512))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    model.add(Dense(512, nb_classes))
    model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adadelta')
    return model

def fit_model(model, X, y, nb_epoch, batch_size, save_weights_file, datagen):

    for e in range(nb_epoch):
        print('Epoch: ', e)
        progbar = Progbar(target=X.shape[0], verbose=True)

        # batch train with realtime data augmentation
        total_accuracy = 0
        total_loss = 0
        current = 0
        for X_batch, y_batch in datagen.flow(X, y, batch_size):

            # prepare the batch with random augmentations
            X_batch, y_batch = batch_warp(X_batch, y_batch)

            # train on the batch
            loss, accuracy = model.train(X_batch, y_batch, accuracy = True)
            
            # update the progress bar
            total_loss += loss * batch_size
            total_accuracy += accuracy * batch_size
            current += batch_size
            if current > X.shape[0]:
                current = X.shape[0]
            else:
                progbar.update(current, [('loss', loss), ('acc.', accuracy)])
        progbar.update(current, [('loss', total_loss/current), ('acc.', total_accuracy/current)])
        
        # checkpoints between epochs
        model.save_weights(save_weights_file, overwrite = True)
    
    return model

def cross_validate(model, X, y, folds, nb_epoch, batch_size, save_weights_file, datagen):
    ''' provides a simple cross validation measurement. It doen't make a new
    model for each fold though, so it isn't actually cross validation... the
    model just gets better with time for now. This is pretty expensive to run. '''

    kf = KFold(X.shape[0], folds)
    scores = []

    for train, test in kf:
        X_train, X_test, y_train, y_test = X[train], X[test], y[train], y[test]

        model = fit_model(model, X_train, y_train, nb_epoch, batch_size, save_weights_file, datagen)
        
        loss, score = model.evaluate(X_test, y_test, show_accuracy=True, verbose=0)
        print ('Loss: ' + str(loss))
        print ('Score: ' + str(score))
        scores.append(score)
    
    scores = np.array(scores)
    print("Accuracy: " + str(scores.mean()) + " (+/- " + str(scores.std()/2) + ")")

def get_predictions(filename, X, y, model, nb_epoch, batch_size, save_weights_file, load_weights_file, load_weights, datagen):
    """ trains and predicts on the mnist data """

    if load_weights:
        model.load_weights(load_weights_file)
    else:
        model = fit_model(model, X, y, nb_epoch, batch_size, save_weights_file, datagen)

    test_data = np.genfromtxt(filename, delimiter=',', skip_header=1, dtype='float32')
    test_data = test_data.reshape(test_data.shape[0], 1, 28, 28)
    test_data /= 255

    return model.predict_classes(test_data, batch_size = batch_size)

def save_predictions(predictions, filename):
    ''' saves the predictions to file in a format that kaggle likes. '''

    predictions_file = open(filename, "wb")
    open_file_object = csv.writer(predictions_file, quotechar='"', quoting=csv.QUOTE_NONNUMERIC)
    open_file_object.writerow(['ImageId','Label'])
    for i in range(0,28000):
        open_file_object.writerow([i+1, predictions[i]])
    predictions_file.close()

def main():

    mode = 'pred' # pred generates submission; test does cross-val; both is both
    folds = 5
    subset = 1 # percent of train file to utilize

    load_weights = False
    load_weights_file = 'weights/pp_3_12.hdf5'
    save_weights_file = 'tmp/checkpoint_weights.hdf5'

    train_file = 'data/train.csv'
    test_file = 'data/test.csv'
    out_file = 'solutions/answers_warp_4_100.csv'

    nb_epoch = 100
    batch_size = 128
    nb_classes = 10

    X = None
    y = None
    datagen = None

    if not load_weights:
        print('loading data...')
        X, y, datagen = load_data(train_file, nb_classes, subset = subset)

    print('building model...')
    model = build_keras(nb_classes)
    
    if mode == 'test' or mode == 'both':
        print('evaluating model...')
        cross_validate(model, X = X, y = y, folds = folds, nb_epoch = nb_epoch, batch_size = batch_size, 
                                            save_weights_file = save_weights_file, datagen = datagen)

    if mode == 'pred' or mode == 'both':
        print('obtaining predictions...')
        save_predictions(get_predictions(test_file, X = X, y = y, 
                                            model = model, nb_epoch = nb_epoch, 
                                            batch_size = batch_size, save_weights_file = save_weights_file,
                                            load_weights_file = load_weights_file, load_weights = load_weights,
                                            datagen = datagen), out_file)
    return 1

if __name__ == '__main__':
    main()
