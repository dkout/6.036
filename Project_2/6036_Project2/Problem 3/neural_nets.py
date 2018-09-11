import numpy as np
import math

"""
 ==================================
 Problem 3: Neural Network Basics
 ==================================
    Generates a neural network with the following architecture:
        Fully connected neural network.
        Input vector takes in two features.
        One hidden layer with three neurons whose activation function is ReLU.
        One output neuron whose activation function is the identity function.
"""


def rectified_linear_unit(x):
    """ Returns the ReLU of x, or the maximum between 0 and x."""
    return max(0,x) 

def rectified_linear_unit_derivative(x):
    """ Returns the derivative of ReLU."""
    if rectified_linear_unit(x)==0:
        return 0
    else:
        return 1 
    
def output_layer_activation(x):
    """ Linear function, returns input as is. """
    return x

def output_layer_activation_derivative(x):
    """ Returns the derivative of a linear function: 1. """
    return 1
        
class Neural_Network():
    """
        Contains the following functions:
            -train: tunes parameters of the neural network based on error obtained from forward propagation.
            -predict: predicts the label of a feature vector based on the class's parameters.
            -train_neural_network: trains a neural network over all the data points for the specified number of epochs during initialization of the class.
            -test_neural_network: uses the parameters specified at the time in order to test that the neural network classifies the points given in testing_points within a margin of error.
    """

    def __init__(self):
        
        # DO NOT CHANGE PARAMETERS
        self.input_to_hidden_weights = np.matrix('1 1; 1 1; 1 1')
        self.hidden_to_output_weights = np.matrix('1 1 1')
        self.biases = np.matrix('0; 0; 0')
        self.learning_rate = .001
        self.epochs_to_train = 10
        self.training_points = [((2,1), 10), ((3,3), 21), ((4,5), 32), ((6, 6), 42)]
        self.testing_points = [(1,1), (2,2), (3,3), (5,5), (10,10)]
        
    def train(self, x1, x2, y):

        ### Forward propagation ###
        input_values = np.matrix([[x1],[x2]]) # 2 by 1

        # Calculate the input and activation of the hidden layer
        z1=np.dot(self.input_to_hidden_weights[0],input_values)[0,0] +self.biases[0,0]
        z2=np.dot(self.input_to_hidden_weights[1],input_values)[0,0]+self.biases[1,0]
        z3=np.dot(self.input_to_hidden_weights[2],input_values)[0,0]+self.biases[2,0]
        # print('z1=',z1,z2,z3)
        hidden_layer_weighted_input = np.matrix([[z1],[z2],[z3]]) #TODO (3 by 1 matrix)
        # print('hidden layer weighted input = ', hidden_layer_weighted_input)
        hidden_layer_activation = np.matrix([[rectified_linear_unit(hidden_layer_weighted_input[0,0])],[rectified_linear_unit(hidden_layer_weighted_input[1,0])],[rectified_linear_unit(hidden_layer_weighted_input[2,0])]])  # TODO (3 by 1 matrix)
        # print('hidden layer activation & hidden to output weights= ', hidden_layer_activation, self.hidden_to_output_weights)
        # print('hidden layer activation = ', hidden_layer_activation)
        output = np.dot(self.hidden_to_output_weights,hidden_layer_activation)[0,0]
        print ('output = ', output)
        activated_output = output # TODO
        
        print("Point:", x1, x2, " Error: ", (0.5)*pow((y - output),2))

        ### Backpropagation ###
        
        # Compute gradients
        output_layer_error = -y+activated_output # TODO
        hidden_layer_error =1   # TODO (3 by 1 matrix)

        bias_gradients = np.transpose(self.hidden_to_output_weights)#   output_layer_error*hidden_to_output_weights
        # print(self.hidden_to_output_weights)

        hidden_to_output_weight_gradients = np.matrix([
            [rectified_linear_unit_derivative(hidden_layer_weighted_input[0,0])],
            [rectified_linear_unit_derivative(hidden_layer_weighted_input[1,0])],
            [rectified_linear_unit_derivative(hidden_layer_weighted_input[2,0])]
        ]) #df/dz # TODO
        # print("Hidden to output weight gradients(df/dz) = ", hidden_to_output_weight_gradients)

        input_to_hidden_weight_gradients = np.matrix([[x1,x2], [x1,x2],[x1,x2]]) #dz/dw TODO
        # print('input_to_hidden_weight_gradients = ', input_to_hidden_weight_gradients)

        # Use gradients to adjust weights and biases using gradient descent
        eta=self.learning_rate
        # print("hidden to output weight gradients and bias gradients = ",hidden_to_output_weight_gradients,bias_gradients)
        self.biases = np.add(self.biases,-eta*output_layer_error*np.multiply(hidden_to_output_weight_gradients,bias_gradients))
        # print('BIAS UPDATED = ', self.biases)
        self.input_to_hidden_weights = np.add(self.input_to_hidden_weights,-eta*output_layer_error*np.multiply(hidden_to_output_weight_gradients, input_to_hidden_weight_gradients))
        # print("New input weights = ", self.input_to_hidden_weights)
        # print('hidden to output weight gradients = ', hidden_to_output_weight_gradients)
        # print('hidden to output weights = ', self.hidden_to_output_weights)
        self.hidden_to_output_weights = np.add(self.hidden_to_output_weights, -eta*output_layer_error*np.transpose(hidden_to_output_weight_gradients)) # TODO
        # print("New hidden to output Weights (V) = ", self.hidden_to_output_weights)
        
    def predict(self, x1, x2):
        input_values = np.matrix([[x1],[x2]]) # 2 by 1

        # Calculate the input and activation of the hidden layer
        z1=np.dot(self.input_to_hidden_weights[0],input_values)[0,0] +self.biases[0,0]
        z2=np.dot(self.input_to_hidden_weights[1],input_values)[0,0]+self.biases[1,0]
        z3=np.dot(self.input_to_hidden_weights[2],input_values)[0,0]+self.biases[2,0]
        # print('z1=',z1,z2,z3)
        hidden_layer_weighted_input = np.matrix([[z1],[z2],[z3]]) #TODO (3 by 1 matrix)
        # print('hidden layer weighted input = ', hidden_layer_weighted_input)
        hidden_layer_activation = np.matrix([[rectified_linear_unit(hidden_layer_weighted_input[0,0])],[rectified_linear_unit(hidden_layer_weighted_input[1,0])],[rectified_linear_unit(hidden_layer_weighted_input[2,0])]])  # TODO (3 by 1 matrix)
        # print('hidden layer activation & hidden to output weights= ', hidden_layer_activation, self.hidden_to_output_weights)
        # print('hidden layer activation = ', hidden_layer_activation)
        output = np.dot(self.hidden_to_output_weights,hidden_layer_activation)[0,0]
        # print ('output = ', output)
        activated_output = output
        
        return activated_output.item()
    
    
    # Run this to train your neural network once you complete the train method
    def train_neural_network(self):
        
        for epoch in range(self.epochs_to_train):
            for x,y in self.training_points:                
                self.train(x[0], x[1], y)
    
    # Run this to test your neural network implementation for correctness after it is trained
    def test_neural_network(self):
        
        for point in self.testing_points:
            print("Point,", point, "Prediction,", self.predict(point[0], point[1]))
            if abs(self.predict(point[0], point[1]) - 7*point[0]) < 0.1:
                print("Test Passed")
            else:
                print("Point ", point[0], point[1], " failed to be predicted correctly.")
                return
        

x = Neural_Network()

x.train_neural_network()

# UNCOMMENT THE LINE BELOW TO TEST YOUR NEURAL NETWORK
x.test_neural_network()  
