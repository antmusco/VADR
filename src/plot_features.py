import sys
import numpy as np
import matplotlib.pyplot as plt

feature_names = ['Borough', 'Num Units', 'Gross Sq.Ft.', 'Year Built', 'Indicated Value', 'Assessment', 'Class']
class_colors = ['r', 'b']

def plot_all_features(file_name):

    """ Generates a cross-validated classifier and plots the training
    error as a function of the specified degree.
    """

    # Load the data from the csv file and count the number of records.
    raw_data = np.loadtxt(file_name, delimiter=',', skiprows=0)

    num_examples = len(raw_data[:, 0])
    num_features = len(raw_data[0, :])

    features = raw_data[:,:-1]
    classes = raw_data[:,-1]
    colors = []

    for i in xrange(num_examples):
    	if classes[i] == 1:
    		colors.append(class_colors[1]) # OVER ASSESSED = BLUE
    	elif classes[i] == -1
    		colors.append((class_colors[0])) # UNDER ASSESSED = RED

    for i in xrange(num_features - 1):
    	for j in xrange(i + 1, num_features - 1):
    		if i == j:
    			continue
    		print i, " ", j
    		feature_1 = features[:, i]
    		feature_2 = features[:, j]
    		plot_features(feature_1, i, feature_2, j, colors)


def plot_features(feature_1, index_1, feature_2, index_2, colors):

	# Plot the errors to the graph.
    plt.scatter(x=feature_1, y=feature_2, color=colors)

    # Show the graph.

    title = feature_names[index_1] + " vs. " + feature_names[index_2]

    plt.title(title)
    plt.xlabel(feature_names[index_1])
    plt.ylabel(feature_names[index_2])
    plt.savefig("./figures/features/" + title + ".png")

# Run from command line.
if __name__ == '__main__':
    data_path = sys.argv[1]
    plot_all_features(data_path)