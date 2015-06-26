#!/usr/bin/env python
# coding: utf-8

from __future__ import absolute_import
from __future__ import print_function

import numpy as np
from scipy.stats import mode

import csv as csv

def load_data(filename):
    """ returns X and y - data and target - as numpy arrays """
    data = np.genfromtxt(filename, delimiter=',', skip_header=1, dtype='float32')
    return data[0::,1]

def tally_votes(votes):
    model = np.zeros(28000)
    resolved = 0
    unresolved = 0

    for i in range(28000):
        modes, counts = mode(votes[i,:])
        if counts[0] != 3:
            print(i, ': ', votes[i,:], ':=',modes[0])
            if counts[0] == 2:
                resolved += 1
            else:
                unresolved += 1
        model[i] = modes[0]
    print('resolved ', resolved, ' disputes...')
    print(unresolved, ' disputes unsolved...')
    return model

def eval_model(model, votes, model_files):
    print('Number of predictions changed per model:')
    for f in range(0, len(model_files)):
        print(model_files[f] + ': ' + str(np.sum(model != votes[:, f])))

def save_predictions(predictions, filename):
    predictions_file = open(filename, "wb")
    open_file_object = csv.writer(predictions_file, quotechar='"', quoting=csv.QUOTE_NONNUMERIC)
    open_file_object.writerow(['ImageId','Label'])
    for i in np.arange(1,28001):
        open_file_object.writerow([i, predictions[i-1]])
    predictions_file.close()

def main():

    model_file = 'solutions/answers_vote.csv'
    model_files = ['solutions/answers_pp_1.csv', 'solutions/answers_pp_2.csv', 'solutions/answers_pp_3.csv']
    votes = np.zeros((28000, len(model_files)))

    print('loading data...')
    for f in range(len(model_files)):
        votes[:, f] = load_data(model_files[f])

    print('voting on model...')
    model = tally_votes(votes)
    print(model)

    print('evaluating model...')
    eval_model(model, votes, model_files)

    print('saving model...')
    save_predictions(model, model_file)

if __name__ == '__main__':
    main()
