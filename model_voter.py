#!/usr/bin/env python
# coding: utf-8

from __future__ import absolute_import
from __future__ import print_function

import numpy as np
from scipy.stats import mode
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib import interactive


import csv as csv

def tally_votes(model_files, data_file, out_file):
    """ saves a model to out_file that is the mode of
    all predictions in the model_files """

    # load all of the models
    votes = np.zeros((28000, len(model_files)))

    for f in range(len(model_files)):
        votes[:, f] = np.genfromtxt(model_files[f], delimiter=',', skip_header=1, dtype='float32')[0::,1]

    # load the test data that it was trained on
    data = np.genfromtxt(data_file, delimiter=',', skip_header=1, dtype='float32')

    interactive(True)

    # vote on the model predictions
    model = np.zeros(28000)
    disputes = 0
    show_imgs = True

    for i in range(28000):
        modes, counts = mode(votes[i,:])

        # if the models weren't unanimous
        if counts[0] != votes.shape[1]:
            print(i, ': ', votes[i,:], ':=',modes[0])
            
            if show_imgs:
                plt.imshow(data[i].reshape((28,28)), cmap='Greys')
                if raw_input('press return to continue or q to end...') == 'q':
                    show_imgs = False

            disputes += 1
        model[i] = modes[0]

    print('Evaluated ', disputes, ' disputes.')
    print('Number of predictions changed per model:')
    for f in range(0, len(model_files)):
        print(model_files[f] + ': ' + str(np.sum(model != votes[:, f])))

    model = model.astype('int')

    # save the new model
    predictions_file = open(out_file, "wb")
    open_file_object = csv.writer(predictions_file, quotechar='"', quoting=csv.QUOTE_NONNUMERIC)
    open_file_object.writerow(['ImageId','Label'])
    for i in range(0,28000):
        open_file_object.writerow([i+1, model[i]])
    predictions_file.close()
    print('File saved to ', predictions_file)

def main():

    out_file = 'solutions/answers_vote_demo.csv'
    data_file = 'data/test.csv'
    model_files = ['solutions/answers_warp_2_100.csv', 'solutions/answers_warp_2_500.csv', 'solutions/answers_warp_4_100.csv']

    model = tally_votes(model_files, data_file, out_file)

if __name__ == '__main__':
    main()
