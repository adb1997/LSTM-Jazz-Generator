# LSTM-Jazz-Generator
Generate your own Jazz music !

Credits: Deeplearning.ai Sequence Models course on Coursera! 

The data has been preprocessed thanks to a function provided by Deeplearning.ai. 

x is a sequence of inputs give to a many-to-many Long Short Term Memory(LSTM) Neural Network.
Each input's length is set as 30 with a probability of having 1/78 values.

The labels y are the same values in x shifted one to the right thereby using the values that have already been generated.

Each LSTM cell as defined through Keras implementation has 2 inputs: 
-Activation
-One-hot encoded vector of input 

Output activation is sent to next LSTM, while the same is fed to a softmax function for output y.

Model built in function DjModel() which is then compiled. The values are instantiated and passed through layer_objects defined outside the function. 

Model is trained for number of epochs (here 100) and then sampled to produce music. 

Note: Create an output directory in root folder where the function will store generated music as midi file. 

Use an online converter for an mp3 version or use an MIDI player. 


