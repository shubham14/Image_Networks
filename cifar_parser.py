# ------ Cifar-10-parser -------------#

import numpy as np
import pandas as pd
import pickle
import os
from sklearn.model_selection import train_test_split

# helper function to load the dataset
def unpickle(file):
	with open(file, 'rb') as fo:
		dict = pickle.load(fo,encoding='latin1')
	return dict 


def data_unpickle(split_ratio):

	data_1 = unpickle(r"C:\Users\Shubham\Desktop\cifar-10-python\data_batch_1")
	data_2 = unpickle(r"C:\Users\Shubham\Desktop\cifar-10-python\data_batch_2")
	data_3 = unpickle(r"C:\Users\Shubham\Desktop\cifar-10-python\data_batch_3")
	data_4 = unpickle(r"C:\Users\Shubham\Desktop\cifar-10-python\data_batch_4")
	data_5 = unpickle(r"C:\Users\Shubham\Desktop\cifar-10-python\data_batch_5")
	
	# merge data and labels of the 5 datasets
	data_list = np.vstack((data_1['data'],\
	data_2['data'],data_3['data'],data_4['data'],\
	data_5['data']))

	labels_list = np.hstack((data_1['labels'],\
	data_2['labels'],data_3['labels'],data_4['labels'],\
	data_5['labels']))

	X_train, X_test, y_train, y_test = train_test_split(data_list, labels_list, test_size=split_ratio, random_state=42)

	return X_train, y_train, X_test, y_test
