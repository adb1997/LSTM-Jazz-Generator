from __future__ import print_function
import IPython
import sys
from music21 import *
import numpy as np
from grammar import *
from qa import *
from preprocess import * 
from music_utils import *
from data_utils import *
from keras.models import load_model, Model
from keras.layers import Dense, Activation, Dropout, Input, LSTM, Reshape, Lambda, RepeatVector
from keras.initializers import glorot_uniform
from keras.utils import to_categorical
from keras.optimizers import Adam
from keras import backend as K

#Checking shape of input dataset 

X, Y, n_values, indices_values = load_music_utils()
print('shape of X:', X.shape)
print('number of training examples:', X.shape[0])
print('Tx (length of sequence):', X.shape[1])
print('total # of unique values:', n_values)
print('Shape of Y:', Y.shape)

n_a = 64 #64 dimensional LSTM hidden states

#Initialise global layer_objects for LSTM layers
reshapor = Reshape((1, 78)) #Keras tensor to reshape input vector               	          
LSTM_cell = LSTM(n_a, return_state = True)  #LSTM cell	       	
densor = Dense(n_values, activation='softmax') #Normal densely connected NN 

def djmodel(Tx, n_a, n_values):
    """
    Implement the model
    
    Arguments:
    Tx -- length of the sequence in a corpus
    n_a -- the number of activations used in our model
    n_values -- number of unique values in the music data 
    
    Returns:
    model -- a keras model with the 
    """
    
    # Defining model with a shape 
    X = Input(shape=(Tx, n_values))
    
    #Defining hidden shapes	
    a0 = Input(shape=(n_a,), name='a0')
    c0 = Input(shape=(n_a,), name='c0')
    a = a0
    c = c0
    
    outputs = []
    
    # Loop over Tx input examples
    for t in range(Tx):
            
        x = Lambda(lambda x: X[:,t,:])(X) #Create Lambda vector to be used as tensor 
		x = reshapor(x) #Calling reshape layer object 
		a, _, c = LSTM_cell(x,initial_state=[a,c]) #Running LSTM cell for one batch of a and c
        out = densor(a) #Passing output through one densely connected NN layer
		outputs.append(out) #append output to list 
        
    
    model = Model(inputs=[X,a0,c0],output=outputs) #initialise model
    
    return model
	
	
model = djmodel(Tx = 30 , n_a = 64, n_values = 78) #Instantiate model		
opt = Adam(lr=0.01, beta_1=0.9, beta_2=0.999, decay=0.01) #Set optimiser

model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy']) #Call compile to get model ready to be trained

#initialise input to model
m = 60
a0 = np.zeros((m, n_a))
c0 = np.zeros((m, n_a))

model.fit([X, a0, c0], list(Y), epochs=100) #fit the model 


""" Sample the model. Taking each a and c from previous cell and forward propagating to generate music"""

def music_inference_model(LSTM_cell, densor, n_values = 78, n_a = 64, Ty = 100):
    """
    Uses the trained "LSTM_cell" and "densor" from model() to generate a sequence of values.
    
    Arguments:
    LSTM_cell -- the trained "LSTM_cell" from model(), Keras layer object
    densor -- the trained "densor" from model(), Keras layer object
    n_values -- integer, umber of unique values
    n_a -- number of units in the LSTM_cell
    Ty -- integer, number of time steps to generate
    
    Returns:
    inference_model -- Keras model instance
    """
    
    # Define the input of the model with a shape 
    x0 = Input(shape=(1, n_values))
    
    # Initial hidden state for the decoder LSTM
    a0 = Input(shape=(n_a,), name='a0')
    c0 = Input(shape=(n_a,), name='c0')
    a = a0
    c = c0
    x = x0

    outputs = []
    
    for t in range(Ty):
        
        a, _, c = LSTM_cell(x, initial_state=[a,c])     
        out = densor(a)
		outputs.append(out)
        x = Lambda(one_hot)(out) #The ouput is saved as a one hot vector to be passed to the next cell 
        
	inference_model = Model(inputs=[x0,a0,c0],output=outputs)
    
    return inference_model
	

inference_model = music_inference_model(LSTM_cell, densor, n_values = 78, n_a = 64, Ty = 50)

#Zero valued initializers for LSTM 
x_initializer = np.zeros((1, 1, 78))
a_initializer = np.zeros((1, n_a))
c_initializer = np.zeros((1, n_a))


def predict_and_sample(inference_model, x_initializer = x_initializer, a_initializer = a_initializer, 
                       c_initializer = c_initializer):
    """
    Predicts the next value of values using the inference model.
    
    Arguments:
    inference_model -- Keras model instance for inference time
    x_initializer -- numpy array of shape (1, 1, 78), one-hot vector initializing the values generation
    a_initializer -- numpy array of shape (1, n_a), initializing the hidden state of the LSTM_cell
    c_initializer -- numpy array of shape (1, n_a), initializing the cell state of the LSTM_cel
    
    Returns:
    results -- numpy-array of shape (Ty, 78), matrix of one-hot vectors representing the values generated
    indices -- numpy-array of shape (Ty, 1), matrix of indices representing the values generated
    """
    
	pred = inference_model.predict([x_initializer,a_initializer,c_initializer])
    #Converting "pred" into an np.array() of indices with the maximum probabilities
    indices = np.argmax(pred,axis=-1)
    # Convert indices to one-hot vectors
    results = to_categorical(indices, num_classes=78)
    
    return results, indices
	

out_stream = generate_music(inference_model)

