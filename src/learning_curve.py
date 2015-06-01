#!/usr/bin/python
# -*- coding: utf-8 -*-

import sys
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn import svm
from sklearn import cross_validation as cv
from sklearn.preprocessing import PolynomialFeatures
from sklearn.cross_validation import StratifiedKFold
from numpy import sign

# Number of folds for cross-validation.
k = 5
training_percentages = [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]

def misclassification_rate(guess, actual):
    return sum(abs(sign(actual) - sign(guess))) / (2.0 * len(actual))


def learning_curve(file_name, classifier_type, degree, kernel, argument):

    """ Calculates the learning curve as a function of the training percentages.
    """

    # Load the data from the csv file and count the number of records.
    raw_data = np.loadtxt(file_name, delimiter=',', skiprows=0)
    n = len(raw_data)

    training_sizes = []

    # Calculate actual training sizes.
    for p in training_percentages:
        training_sizes.append(round(n * p))

    print training_sizes

    # Create new classifier.
    classifier = 0
    param_type = 0
    if classifier_type == "Lasso":
        param_type = "alpha"
        classifier = linear_model.Lasso(alpha=argument)
    elif classifier_type == "Ridge":
        param_type = "alpha"
        classifier = linear_model.Ridge(alpha=argument)
    elif classifier_type == "SVC":
        param_type = "gamma"
        classifier = svm.SVC(kernel=kernel, gamma=argument)
    else:
        print_help()
        return

    # Trasform the data. include_bias=false since we don't need to add an
    #  additional dummy variable.
    poly = PolynomialFeatures(degree=degree, include_bias=True)
    
    # X is the transformed features matrix.
    x = poly.fit_transform(raw_data[:,:-1])
    y = raw_data[:, -1]

    # Keep track of the errors.
    errors = np.zeros(len(training_sizes))

    # Test all training sizes.
    for s in xrange(len(training_sizes)):

        # Split the features from the classes.
        train_x = x[:training_sizes[s],:]
        train_y = y[:training_sizes[s]]
        test_x = x[training_sizes[s]:,:]
        test_y = y[training_sizes[s]:]

        # Build the classifier by fitting it to the training data.
        classifier.fit(train_x, train_y)

        # Predict the test data using the classifier.
        results = classifier.predict(test_x)

        # Add the misclassification rate for this degree/alpha pair.
        errors[s] = misclassification_rate(results, test_y)

    if classifier_type == "SVC":
        classifier_type = classifier_type + "-" + kernal

    # Plot the misclassification rate as a function of alpha for all degrees.
    plot_errors(training_sizes, errors, classifier_type + " Learning Curve - " + \
        param_type + " " +str(argument) , classifier_type, argument)


def plot_errors(x, y, title, classifier_type, argument):

    """ Plot the test error as a function of the number of training examples and save to './figures'. """

    # Plot the errors to the graph.
    plt.plot(x, y)

    # Show the graph.
    plt.title(title)
    plt.ylabel('Misclassification Rate')
    plt.xlabel('Training Sizes')
    plt.savefig("./figures/" + classifier_type + "_" + str(argument) + "_learning_curve.png")
    plt.show()


def print_help():

    """ Print the usage of learning_curve.py to the console. """

    print "Calculates the learning curve for the given classifier with parameters and"
    print "graphs the results as a function of the training size."
    print "Usage: python learning_curve.py [data path] [classifier type] [degree] [kernel] [arg]"
    print "\tclassifier type = 'Lasso', 'Ridge', 'SVC'"
    print "\tkernel (only for SVC) = 'none', 'rbf', 'linear'"


# Run from command line.
if __name__ == '__main__':
    if len(sys.argv) < 6:
        print_help()
    else:
        data_path = sys.argv[1]
        selection = sys.argv[2]
        degree = int(sys.argv[3])
        kernal = sys.argv[4]
        argument = float(sys.argv[5])
        learning_curve(data_path, selection, degree, kernal, argument)
