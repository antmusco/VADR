TITLE:
	Valuation/Assesment Discrepancy

AUTHORS:
	Anthony Musco
	Matthew Del Signore
	Fumi Honda
	Gustavo Poscidonio


DIRECTORY:
	Python source files located in 'src' folder. Data files in 'data' folder. Plot figures
	in the 'figures' folder.


DATA:
	The data files included in the 'data' folder are 'training_data_norm.csv' and
	'test_data_norm.csv'. These file contains the data after subtracting the mean and 
	dividing by the standard deviation for each feature. The source data was also 
	included as an .xlsx file.

	In case it is needed, the mean/standard deviation information has been saved in 
	'training_data_mean_sd.csv'


FIGURES:
	The figures (thus far) plot the error percentage as a function of the number of 
	training examples for degree 1, 2, 3, and 5 degree polynomials.


CODE:

	1. LEARN.PY
	The python file 'src/learn.py' contains the primary code for testing different
	classifiers with different parameters. The script can be run from the command line
	like this:

		1. Navigate to CSE-353/

		2. To use a linear classifier:
			
			python ./src/learn.py ./data/training_data_norm.csv linear

		3. Use a lasso classifier with alpha of 0.01

			python ./src/learn.py ./data/training_data_norm.csv lasso 0.01

		4. Use a svc with radial basis function and gamma value of 0.01

			python ./src/learn.py ./data/training_data_norm.csv svc rbf 0.01

		5. In general, the format is this:

			python ./src/learn.py [data to be learned] [classifier to use] [other params]

		6. For more info, type:

			python ./src/learn.py help

	After running the script, a plot of the results will be saved to the 'figures'
	folder with the name of the classifier and any parameters used. Use this script
	to test different functions with varying parameters to learn the optimal classifier.

	2. TEST_PARAMS.PY
	The python file 'test_params.py' contains the primary code for generating plots
	for classifiers with different parameters. The parameters are hard-coded into 
	the file and must be changed manually.

		1. Navigate to CSE-353/

		2. To use a 'lasso classifier:
			
			python ./src/test_params.py ./data/training_data_norm.csv lasso

	3. TRAIN_AND_TEST.PY
	The python file 'train_and_test.py' is the primary testing program. It takes two
	arguments: the training data file and the test data file. Type:

		python ./src/train_and_test.py ./data/training_data_norm.csv ./data/test_data_norm.csv lasso 0.001

	To run a lasso classifier with alpha parameter of 0.001.