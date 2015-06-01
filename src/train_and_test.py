#!/usr/bin/python
# -*- coding: utf-8 -*-

import sys
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn import svm
from sklearn.preprocessing import PolynomialFeatures
from numpy import sign

# Test differnt thresholds
thresholds = np.arange(0.0, 1.0, 0.005)

def misclassification_rate(guess, actual, threshold):
    miss_total = 0.0
    num_classified = 0.0
    total_examples= len(guess)
    for i in xrange(total_examples):
        if abs(guess[i]) > threshold:
            miss_total += abs(sign(actual[i]) - sign(guess[i])) / 2.0
            num_classified += 1.0
    return {'error':miss_total / num_classified, 'count': num_classified / total_examples,\
        'score': miss_total / (num_classified - miss_total)}

def train_and_test(train_file_name, test_file_name, classifier_type, degree, arg):

    """ Generates a classifier and applies it to test data. """

    # Load the data from the csv file and count the number of records.
    training_data = np.loadtxt(train_file_name, delimiter=',', skiprows=0)
    test_data = np.loadtxt(test_file_name, delimiter=',', skiprows=0)

    # Trasform the data. include_bias=false since we don't need to add an
    #  additional dummy variable.
    poly = PolynomialFeatures(degree=degree, include_bias=True)

    # X is the transformed features matrix.
    train_x = training_data[:, :-1]
    test_x = test_data[:, :-1]
    train_x = poly.fit_transform(train_x)
    test_x = poly.fit_transform(test_x) 

    # Y is the list of classifications.
    train_y = training_data[:,-1]
    test_y = test_data[:,-1]

    # Create new classifier.
    classifier = 0
    if classifier_type == "Lasso":
        classifier = linear_model.Lasso(alpha=arg)
    elif classifier_type == "Ridge":
        classifier = linear_model.Ridge(alpha=arg)
    elif classifier_type == "SVC-RBF":
        classifier = svm.SVC(kernel='rbf', gamma=arg)
    elif classifier_type == "SVC-Linear":
        classifier = svm.SVC(kernel='linear')

    # Build the classifier by fitting it to the training data.
    classifier.fit(train_x, train_y)

    results = classifier.predict(test_x)

    # Calculate the misclassification rate for this degree/alpha pair.
    errors = np.zeros((3, len(thresholds)))
    for t in xrange(len(thresholds)):
        err = misclassification_rate(results, test_y, thresholds[t])
        errors[0, t] = err['error']
        errors[1, t] = err['count']
        errors[2, t] = err['score']

    plot_errors(thresholds, errors, "Threshold Performance - " + classifier_type + "-" + str(arg), classifier_type, arg)


def plot_errors(x, y, title, classifier_type, arg):

    """ Plot the test error as a function of the number of training examples and save to './figures'. """

    # Plot the errors to the graph.
    plt.plot(x, y[0, :], color='r', label='Misclassification Rate')
    plt.plot(x, y[1, :], color='b', label='Classification Percentage')
    plt.plot(x, y[2, :], color='g', label='Threshold Score')

    # Show the graph.
    plt.title(title)
    plt.legend()
    plt.ylabel('Rate')
    plt.xlabel('Thresholds')
    plt.savefig("./figures/" + classifier_type + "_" + str(arg) + "_threshold.png")
    plt.show()


def print_help():

    """ Print the usage of lasso_classifier.py to the console. """

    print "Usage: [train data path] [test data path] [classifier type] [degree] [alpha/gamma]"
    print "\ttype = 'Lasso', 'Ridge', 'SVC-RBF', 'SVC-Linear'"


# Run from command line.
if __name__ == '__main__':
    if len(sys.argv) < 6:
        print_help()
    else:
        train_path = sys.argv[1]
        test_path = sys.argv[2]
        selection = sys.argv[3]
        degree = int(sys.argv[4])
        arg = float(sys.argv[5])
        train_and_test(train_path, test_path, selection, degree, arg)
