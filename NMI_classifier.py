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
train_x, test_x, train_y, test_y = train_test_split(X,Y,test_size = 0.2,random_state = 42)

# Set learning parameters
learning_rate = 0.3
learning_epochs = 1000
cost_history = np.empty(shape = [1],dtype = float)
n_dim = X.shape[1]
n_class = 2

n_hidden_1 = 60
n_hidden_2 = 60
n_hidden_3 = 60
n_hidden_4 = 60

W = tf.Variable(tf.zeros([n_dim,n_class]))
b = tf.Variable(tf.zeros([n_class]))
x = tf.placeholder(tf.float32,[None,n_dim])
y_ = tf.placeholder(tf.float32,[None,n_class])

# Define the model
def multilayer_perceptron(x,weights,biases):
	layer_1 = tf.add(tf.matmul(x,weights['h1']),biases['b1'])
	layer_1 = tf.nn.relu(layer_1)

	layer_2 = tf.add(tf.matmul(layer_1,weights['h2']),biases['b2'])
	layer_2 = tf.nn.sigmoid(layer_2)

	layer_3 = tf.add(tf.matmul(layer_2,weights['h3']),biases['b3'])
	layer_3 = tf.nn.sigmoid(layer_3)

	layer_4 = tf.add(tf.matmul(layer_3,weights['h4']),biases['b4'])
	layer_4 = tf.nn.relu(layer_4)

	out_layer = tf.add(tf.matmul(layer_4,weights['out']),biases['out'])
	return out_layer



