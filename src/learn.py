#!/usr/bin/python
# -*- coding: utf-8 -*-

import sys
import numpy as np
from numpy import sign
import matplotlib.pyplot as plt
from sklearn.cross_validation import StratifiedKFold
from sklearn.preprocessing import PolynomialFeatures
from sklearn import cross_validation as cv
from sklearn import linear_model
from sklearn import svm

degrees = [1, 2, 3, 5, 7]
colors = ['r', 'g', 'b', 'm', 'c']


def test_classifier(data_file, classifier):

    """ Tests the classifier on the data supplied in csv format.

    Applies k-fold cross-validation on n/16, n/8, n/4, n/2, and n training examples
    for various degrees. The learning curve is plotted as a graph indicating the
    classification error vs. number of examples for each degree used.

    Keyword arguments:
    data_file  -- the data supplied in a comma-separated-value file. It is assumed to
                  be formatted with 'm' records and 'n - 1' features, with the 'n'th
                  column being the labels of the class.
    classifier -- the classifier to be tested. Can be any of the classifiers provided
                  by the sci-kit learn API (e.g. linear_model.LinearRegression(),
                  svm.SVC(), etc.)

    """

    # Load the data from the csv file and count the number of records.
    raw_data = np.loadtxt(data_file, delimiter=',', skiprows=0)
    n = len(raw_data)

    # Initialize all of the variable parameters.
    data_size = [n / 16, n / 8, n / 4, n / 2, n]
    for i in xrange(len(data_size)):
        data_size[i] = round(data_size[i] / 5.0) * 5.0

    k = 5
    errors = np.zeros((len(data_size), len(degrees)))

    # Transform all of the data and cross-validate.
    for d in xrange(len(degrees)):
        for s in xrange(len(data_size)):

            # Get the training and test data.
            x = raw_data[:data_size[s], :-1]
            y = raw_data[:data_size[s], -1]

            # Transform the data.
            poly = PolynomialFeatures(degree=degrees[d], include_bias=True)
            x = poly.fit_transform(x)
            skf = cv.StratifiedKFold(y, n_folds=k)

            # Perform k-fold cross-validation.
            local_errors = []
            for train_index, test_index in skf:
                # Fit the data.
                classifier.fit(x[train_index], y[train_index])
                # Predict the test data.
                results = classifier.predict(x[test_index])
                # Calculate errors.
                misclassifications = sum(abs(y[test_index] - sign(results)) / 2.0)
                local_errors.append(misclassifications / len(test_index))
                # print "Misclassifications: ", misclassifications, " / ", len(test_index)
                # local_errors.append((1.0 - (sum(y[test_index] * sign(results)) / float(len(train_index)))) / 2.0)


            # Take the average over all folds.
            errors[s, d] = sum(local_errors) / len(local_errors)

    # Return the errors.
    return {'errors': errors, 'data_size': data_size}


def print_help():

    """ Print the usage of learn.py to the console. """

    print "Usage: "
    print "\tlinear [] = linear classifier."
    print "\tlasso [alpha] = lasso classifier with alpha value."
    print "\tridge [alpha] = ridge classifier with alpha value."
    print "\tsvc [kernel] [gamma] [degree] = support-vector classifier with specified kernal, gamma, and degree."


def plot_errors(errors, data_size, title):

    """ Plot the test error as a function of the number of training examples and save to './figures'. """

    # Plot the errors to the graph.
    for i in xrange(len(degrees)):
        plt.plot(data_size, errors[:, i], color=colors[i], label=degrees[i])
    # Show the graph.
    plt.title(title)
    plt.legend()
    plt.ylabel('Error Percentage')
    plt.xlabel('Number of Examples')
    plt.savefig("./figures/" + title + ".png")
    plt.show()


def linear_regression_classifier(file_path):

    """ Generate the learning curve for a linear classifier on the specified data. """

    result = test_classifier(file_path, linear_model.LinearRegression())
    plot_errors(result['errors'], result['data_size'], 'Linear_Classifier__')


def lasso_classifier(file_path, alpha):

    """ Generate the learning curve for a LASSO classifier on the specified data with the given alpha. """

    result = test_classifier(file_path, linear_model.Lasso(max_iter=10000, alpha=alpha))
    plot_errors(result['errors'], result['data_size'], 'Lasso_Classifier__alpha-' + str(alpha))


def ridge_regression_classifier(file_path, alpha):

    """ Generate the learning curve for a ridge classifier on the specified data with the given alpha. """
    #linear regression is subset of Ridge and Lasso
    result = test_classifier(file_path, linear_model.Ridge(alpha=alpha))
    plot_errors(result['errors'], result['data_size'], 'Ridge_Classifier__alpha-' + str(alpha))


def support_vector_classifier(file_path, kernel, gamma):

    """ Generate the learning curve for a svc classifier on the specified data with the given kernel and gamma. """

    result = test_classifier(file_path, svm.SVC(kernel=kernel, gamma=gamma, degree=3))
    plot_errors(result['errors'], result['data_size'], 'SVM_Classifier__kernel-' + kernel + '__gamma-' + str(gamma))

def bayesian_ridge_classifier(file_path, alpha_1, alpha_2, lambda_1, lambda_2):

    """ Generate the learning curve for a bayesian r classifier on the specified data with the given parameters. """

    result = test_classifier(file_path, linear_model.BayesianRidge(alpha_1=alpha_1, alpha_2=alpha_2, lambda_1=lambda_1, lambda_2=lambda_2))
    plot_errors(result['errors'], result['data_size'], 'BayesianRidge__alpha1-' + str(alpha_1) + '_alpha2-' + str(alpha_2) + \
        '_lambda1-' + str(lambda_1) + '_lambda2-' + str(lambda_2))

def perceptron_classifier(file_path, alpha):

    """ Generate the learing curve for a perceptron classifier on the specified data with the given alpha. """

    result = test_classifier(file_path, linear_model.Perceptron(penalty='l2', alpha=alpha))
    plot_errors(result['errors'], result['data_size'], 'Perceptron_Classifier__alpha-' + str(alpha))

# Run from command line.
if __name__ == '__main__':
    if len(sys.argv) < 3:
        print_help()
    else:
        data_path = sys.argv[1]
        selection = sys.argv[2]
        if selection == "help":
            print_help()
        elif selection == "linear":
            if len(sys.argv) > 3:
                print_help()
            else:
                linear_regression_classifier(data_path)
        elif selection == "lasso":
            if len(sys.argv) > 4:
                print_help()
            else:
                lasso_classifier(data_path, float(sys.argv[3]))
        elif selection == "ridge":
            if len(sys.argv) > 4:
                print_help()
            else:
                ridge_regression_classifier(data_path, float(sys.argv[3]))
        elif selection == "perceptron":
            if len(sys.argv) > 4:
                print_help()
            else:
                perceptron_classifier(data_path, float(sys.argv[3]))
        elif selection == "svc":
            if len(sys.argv) > 6:
                print_help()
            else:
                support_vector_classifier(data_path, sys.argv[3], float(sys.argv[4]))
        elif selection == "bayes":
            if len(sys.argv) > 8:
                print_help()
            else:
                bayesian_ridge_classifier(data_path, float(sys.argv[3]), float(sys.argv[4]), float(sys.argv[5]), float(sys.argv[6]))
