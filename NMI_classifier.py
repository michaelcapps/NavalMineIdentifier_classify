import matplotlib.pyplot as plt 
import tensorflow as tf
import numpy as np 
import pandas as pd 
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

def read_dataset(dataset):
	#read into a dataframe
	df = pd.read_csv(dataset)
	X = df[df.columns[0:60]].values #features
	y = df[df.columns[60]] #labels

	# one hot encoding
	encoder = LabelEncoder()
	encoder.fit(y)
	y = encoder.transform(y)
	Y = one_hot_encode(y)
	
	return [X,Y]

def one_hot_encode(labels):
	n_labels = len(labels)
	n_unique_labels = len(np.unique(labels))
	one_hot_encode = np.zeros((n_labels,n_unique_labels))
	one_hot_encode[np.arange(n_labels),labels] = 1
	return one_hot_encode


#### Read in and set up the data ####
dataset = 'sonar.all-data.csv'
X,Y = read_dataset(dataset)
X,Y = shuffle(X,Y,random_state = 1)




