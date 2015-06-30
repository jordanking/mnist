#!/usr/bin/env python
# coding: utf-8

from __future__ import absolute_import
from __future__ import print_function

import numpy as np
from scipy.stats import mode

import csv as csv

def tally_votes(model_files, out_file):
    """ saves a model to out_file that is the mode of
    all predictions in the model_files """

    # load all of the models
    votes = np.zeros((28000, len(model_files)))

    for f in range(len(model_files)):
        votes[:, f] = np.genfromtxt(model_files[f], delimiter=',', skip_header=1, dtype='float32')[0::,1]

    # vote on the model predictions
    model = np.zeros(28000)
    disputes = 0

    for i in range(28000):
        modes, counts = mode(votes[i,:])

        # if the models weren't unanimous
        if counts[0] != votes.shape[1]:
            print(i, ': ', votes[i,:], ':=',modes[0])
            disputes += 1
        model[i] = modes[0]

    print('Resolved ', disputes, ' disputes...')
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

def main():

    out_file = 'solutions/answers_vote.csv'
    model_files = ['solutions/answers_pp_600.csv', 'solutions/answers_warp_1_45.csv', 'solutions/answers_warp_2_45.csv']

    model = tally_votes(model_files, out_file)

if __name__ == '__main__':
    main()
