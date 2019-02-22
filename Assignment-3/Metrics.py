import numpy as np

def accuracy_binary(pred , Y):
	'''
	Y is binary column vector
	'''
	pred =pred > 0.5
	acc = np.sum(pred == Y)
	acc = float(acc) / len(Y)
	return acc

def accuracy_multiclass(pred, Y):
	'''
	Y is column vector of one hot row vectors
	'''
	acc = np.sum(pred.argmax(1)==Y.argmax(1))
	acc = float(acc) / len(Y)
	return accs

def get(identifier):
	if identifier == None:
		return accuracy_binary
	elif identifier == 'accuracy_binary':
		return accuracy_binary
	elif identifier == 'accuracy_multiclass':
		return accuracy_multiclass
