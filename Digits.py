#!/usr/bin/env python
# coding: utf-8

# author: Jordan King

from __future__ import absolute_import
from __future__ import print_function

import numpy as np

import csv as csv

from sklearn.cross_validation import KFold

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras.preprocessing.image import ImageDataGenerator
from keras.layers.advanced_activations import PReLU
from keras.callbacks import ModelCheckpoint


def load_data(filename):
    """ returns X and y - data and target - as numpy arrays """
    data = np.genfromtxt(filename, delimiter=',', skip_header=1, dtype='float32')
    return [data[0::,1::], data[0::,0]]

def preprocess_data(X, y, nb_classes, fancy):
    """ returns X and y as shaped for keras and normalized to [0,1]
        plus any fancy image preprocessing selected. """

    X = X.reshape(X.shape[0], 1, 28, 28)
    X /= 255

    y = np_utils.to_categorical(y, nb_classes)

    if fancy:
        datagen = ImageDataGenerator(featurewise_center=False,
                                        samplewise_center=False,
                                        featurewise_std_normalization=False,
                                        samplewise_std_normalization=False,
                                        zca_whitening=False,
                                        rotation_range=4,
                                        width_shift_range=0.08,
                                        height_shift_range=0.08,
                                        horizontal_flip=False,
                                        vertical_flip=False)
        datagen.fit(X)

        flow = datagen.flow(X, y, batch_size = len(y))

        Xt, yt = flow.next()
        np.savetxt('transx.out', Xt.reshape(Xt.shape[0], 784))

        X = np.append(X, Xt, axis = 0)
        y = np.append(y, yt, axis = 0)


    return [X, y]

def build_keras(nb_classes):

    model = Sequential()

    model.add(Convolution2D(32, 1, 3, 3, border_mode='full')) 
    model.add(Activation('relu'))

    model.add(Convolution2D(32, 32, 3, 3))
    model.add(Activation('relu'))

    model.add(MaxPooling2D(poolsize=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())

    model.add(Dense(32*196, 128))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    model.add(Dense(128, nb_classes))
    model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adadelta')
    return model

def cross_validate(model, X, y, folds, nb_epoch, batch_size):

    kf = KFold(X.shape[0], folds)
    scores = []

    for train, test in kf:
        X_train, X_test, y_train, y_test = X[train], X[test], y[train], y[test]
    
        model.fit(X_train,y_train,
                  batch_size=batch_size, 
                  nb_epoch=nb_epoch, 
                  show_accuracy=True, 
                  verbose=1, 
                  validation_data=(X_test, y_test))
        
        loss, score = model.evaluate(X_test, y_test, show_accuracy=True, verbose=0)
        print ('Loss: ' + str(loss))
        print ('Score: ' + str(score))
        scores.append(score)
    
    scores = np.array(scores)
    print("Accuracy: " + str(scores.mean()) + " (+/- " + str(scores.std()/2) + ")")
    return scores

def get_predictions(filename, X, y, model, nb_epoch, batch_size, save_weights_file, load_weights_file, load_weights):
    """ returns a numpy array with predictions for the test file """

    if load_weights:
        model.load_weights(load_weights_file)
    else:
        checkpointer = ModelCheckpoint(filepath=save_weights_file, verbose=1, save_best_only=True)

        model.fit(X,y,
                  batch_size=batch_size, 
                  nb_epoch=nb_epoch, 
                  show_accuracy=True, 
                  verbose=1, 
                  callbacks=[checkpointer])

    test_data = np.genfromtxt(filename, delimiter=',', skip_header=1, dtype='float32')
    test_data = test_data.reshape(test_data.shape[0], 1, 28, 28)
    test_data /= 255

    return model.predict_classes(test_data, batch_size = batch_size)

def save_predictions(predictions, filename):
    predictions_file = open(filename, "wb")
    open_file_object = csv.writer(predictions_file, quotechar='"', quoting=csv.QUOTE_NONNUMERIC)
    open_file_object.writerow(['ImageId','Label'])
    for i in np.arange(1,28001):
        open_file_object.writerow([i, predictions[i-1]])
    predictions_file.close()

def main():

    mode = 'pred'
    fancy_preprocess = True
    load_weights = False
    load_weights_file = 'weights/pp_3_12.hdf5'
    save_weights_file = 'tmp/checkpoint_weights_big.hdf5'
    train_file = 'data/train.csv'
    test_file = 'data/test.csv'
    out_file = 'solutions/answers_pp_big.csv'
    nb_epoch = 600
    folds = 5
    batch_size = 128
    nb_classes = 10

    X = None
    y = None

    if not load_weights:
        print('loading data...')
        X, y = load_data(train_file)

        print('preprocessing data...')
        X, y = preprocess_data(X, y, nb_classes, fancy_preprocess)

    print('building model...')
    model = build_keras(nb_classes)
    
    if mode == 'test' or mode == 'both':
        print('evaluating model...')
        cross_validate(model, X = X, y = y, folds = folds, nb_epoch = nb_epoch, batch_size = batch_size)
    if mode == 'pred' or mode == 'both':
        print('obtaining predictions...')
        save_predictions(get_predictions(test_file, X = X, y = y, 
                                            model = model, nb_epoch = nb_epoch, 
                                            batch_size = batch_size, save_weights_file = save_weights_file,
                                            load_weights_file = load_weights_file, load_weights = load_weights), out_file)

if __name__ == '__main__':
    main()
