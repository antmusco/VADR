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


# The degrees of the polynomial features to be tested.
degrees = [1, 3, 5]
# Colors of each degree.
colors = ['r', 'g', 'b', 'm', 'y', 'c']
# Number of folds for cross-validation.
k = 5
# Alphas to be tested for the classifier.
alphas_lasso = [1e3, 1e2, 1e1, 1e0, 1e-1, 1e-2, 1e-3, 1e-4, 1e-6, 1e-8]
alphas_ridge = [1e5, 1e4, 1e3, 1e2, 1e1, 1e0, 1e-1, 1e-2, 1e-3, 1e-4]
lambdas_bayes = [1e5, 1e4, 1e3, 1e2, 1e1, 1e0, 1e-1, 1e-2]
gammas = [1e3, 1e2, 1e1, 1e0, 1e-1, 1e-2, 1e-3, 1e-4, 1e-6, 1e-8]


def misclassification_rate(guess, actual):
    return sum(abs(sign(actual) - sign(guess))) / (2.0 * len(actual))


def test_params(file_name, classifier_type):

    """ Generates a cross-validated classifier and plots the training
    error as a function of the specified degree.
    """

    # Load the data from the csv file and count the number of records.
    raw_data = np.loadtxt(file_name, delimiter=',', skiprows=0)

    # Y is the list of classifications.
    y = raw_data[:,-1]

    args = 0

    # Create new classifier.
    classifier = 0
    if classifier_type == "Lasso":
        classifier = linear_model.Lasso()
        args = alphas_lasso
    elif classifier_type == "Ridge":
        classifier = linear_model.Ridge()
        args = alphas_ridge
    elif classifier_type == "SVC-RBF":
        classifier = svm.SVC(kernel='rbf')
        args = gammas
    elif classifier_type == "SVC-Linear":
        classifier = svm.SVC(kernel='linear')
        args = gammas
    elif classifier_type == "Bayes":
        classifier = linear_model.BayesianRidge();
        args = lambdas_bayes

    # errors = a 2 dimensial array:
    #   rows = Degree if the transformed features.
    #   columns = parameter value.
    errors = np.zeros((len(degrees), len(args)))

    # Calculate the misclassification rate for each degrees. 
    for d in xrange(len(degrees)):

        # Trasform the data. include_bias=false since we don't need to add an
        #  additional dummy variable.
        poly = PolynomialFeatures(degree=degrees[d], include_bias=True)
        
        # X is the transformed features matrix.
        x = poly.fit_transform(raw_data[:,:-1])

        # Stratified K Fold will partition the data into train and test
        #   indices to perform cross validation.
        skf = cv.StratifiedKFold(y, n_folds=k)

        # Calculate the misclassification rate for each parameter value.
        for a in xrange(len(args)):

            # Update the alpha for the classifier.
            if classifier_type == "Lasso" or classifier_type == "Ridge":
                classifier.set_params(alpha=args[a])
            elif classifier_type == "SVC-RBF":
                classifier.set_params(gamma=args[a])
            elif classifier_type == "Bayes":
                classifier.set_params(lambda_1 = lambdas_bayes[a], lambda_2 = lambdas_bayes[a])

            # Perform k-fold cross-validation.
            for train_index, test_index in skf:

                # Build the classifier by fitting it to the training data.
                classifier.fit(x[train_index], y[train_index])

                # Predict the test data using the classifier.
                results = classifier.predict(x[test_index])

                # Add the misclassification rate for this degree/alpha pair.
                errors[d, a] += misclassification_rate(results, y[test_index])

            # Take the average over all folds.
            errors[d, a] /= len(skf)

    x_axis = 0
    if classifier_type == "Lasso" or classifier_type == "Ridge":
        x_axis = "Alpha"
    else:
        x_axis = "Gamma"

    # Plot the misclassification rate as a function of alpha for all degrees.
    plot_errors(args, errors, classifier_type + " Misclassification Error vs. " + x_axis, classifier_type)


def plot_errors(x, y, title, classifier_type):

    """ Plot the test error as a function of the number of training examples and save to './figures'. """

    # Plot the errors to the graph.
    for i in xrange(len(y)):
        plt.plot(x, y[i, :], color=colors[i], label=degrees[i])

    # Determine the x-axis title.
    x_axis = 0
    if classifier_type == "Lasso" or classifier_type == "Ridge":
        x_axis = "Alpha"
    else:
        x_axis = "Gamma"

    # Show the graph.
    plt.title(title)
    plt.legend()
    plt.ylabel('Misclassification Error')
    plt.xlabel(x_axis)
    plt.xscale('log')
    plt.savefig("./figures/" + classifier_type + "_test_params.png")
    plt.show()


def print_help():

    """ Print the usage of lasso_classifier.py to the console. """

    print "Usage: [data] [type]"
    print "\ttype = 'lasso', 'ridge', 'svc-rbf', 'svc-linear'"


# Run from command line.
if __name__ == '__main__':
    if len(sys.argv) < 3:
        print_help()
    else:
        data_path = sys.argv[1]
        selection = sys.argv[2]
        test_params(data_path, selection)
