import numpy as np

"""
This file defines various metrics to check the correctness of the model
the function are defined in the following format:-
(Note that any function can be added as long as it follows the format)


def metric_name(pred , Y):
	any operations
	return accuracy

Inputs
-pred: a numpy array reprsenting predictions made by the model
-Y : a numpy  array representing the target variable

Note that shape(pred) = shape(Y)

Returns
- accuracy : a scalar value to represent the correctness

Note that you will need to include the new metric function string
to the get function.
"""

def accuracy_binary(pred , Y):
	"""
	Computes the zero one accuracy for a binary classificatiion

	Inputs:
	-pred: Column vector representing probabilities of Class == 1
	-Y: Binary column vector
	"""
	pred =pred > 0.5
	acc = np.sum(pred == Y)
	acc = float(acc) / len(Y)
	return acc

def accuracy_multiclass(pred, Y):
	"""
	Computes the zero one accuracy for a Multiclass classificatiion

	Inputs:
	-pred: Column vector representing probabilities of Class == 1
	-Y: Column vector of one hot row vectors
	"""
	#print("acc, pred = ",pred)
	#print("acc, Y = ",Y)
	acc = np.sum(pred.argmax(1)==Y.argmax(1))
	acc = float(acc) / len(Y)
	return acc

def get(identifier):
	"""
    This function gets fetches the metric identified by
    the string 'identifier'.If such a function is not implemented
    this raises an Exception.


    Inputs:
    - identifier : a string to identify the metric function to fetch
    (default value = None.This fetches accuracy_binary)

    """
	if identifier == None:
		return accuracy_binary
	elif identifier == 'accuracy_binary':
		return accuracy_binary
	elif identifier == 'accuracy_multiclass':
		return accuracy_multiclass
	else:
		raise Exception('The {} metric is not implemented'.format(identifier))
