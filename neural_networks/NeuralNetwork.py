import time
import random
import numpy as np
from utils import *
from transfer_functions import * 


class NeuralNetwork(object):
    
    def __init__(self, input_layer_size, hidden_layer_size, output_layer_size, iterations=50, learning_rate = 0.1):
        """
        input: number of input neurons
        hidden: number of hidden neurons
        output: number of output neurons
        iterations: how many iterations
        learning_rate: initial learning rate
        """
       
        # initialize parameters
        self.iterations = iterations   #iterations
        self.learning_rate = learning_rate
     
        
        # initialize arrays
        self.input = input_layer_size+1  # +1 for the bias node in the input Layer
        self.hidden = hidden_layer_size+1 #+1 for the bias node in the hidden layer 
        self.output = output_layer_size

        # set up array of 1s for activations
        self.a_input = np.ones(self.input)
        self.a_hidden = np.ones(self.hidden)
        self.a_out = np.ones(self.output)
        
        # set up array of 1s for outputs
        self.o_hidden = np.ones(self.hidden)
        self.o_out = np.ones(self.output)
        
        #create randomized weights Yann Lecun method in 1988's paper ( Default values)
        input_range = 1.0 / self.input ** (1/2)
        self.W_input_to_hidden = np.random.normal(loc = 0, scale = input_range, size =(self.input, self.hidden-1))
        self.W_hidden_to_output = np.random.uniform(size = (self.hidden, self.output)) / np.sqrt(self.hidden)
        
        # set up array containing the error
        self.errors = np.ones(self.output)
        
    def weights_initialisation(self,wi,wo):
        self.W_input_to_hidden=wi # weights between input and hidden layers
        self.W_hidden_to_output=wo # weights between hidden and output layers
   

       
        
    #========================Begin implementation section 1============================================="    
    def sigmoid(x):
        return 1.0 / (1.0 + math.exp(-x))
    
    def feedForward(self, inputs):
        v_sigmoid = np.vectorize(sigmoid)
        
        # Compute input activations
        inputs = np.append(inputs, 1)
        self.a_input = np.multiply(self.a_input, inputs)
        print('Inputs', self.a_input)
        
        #Compute  hidden activations
        a_hidden_without_bias = self.W_input_to_hidden.T.dot(self.a_input)
        activation_dummy = 0
        # add value of one to the vector above
        self.a_hidden = np.append(a_hidden_without_bias, activation_dummy)
        print('Activation Hidden', self.a_hidden)
        
        # Compute the function output of the hidden layer
        self.o_hidden = v_sigmoid(self.a_hidden)
        self.o_hidden[-1] = 1
        
        print('Output Hidden', self.o_hidden)
        
        # Compute output activations
        self.a_output = self.W_hidden_to_output.T.dot(self.o_hidden)
        self.o_output = v_sigmoid(self.a_output)
        print('Activation Output', self.a_output)
        print('Output Output', self.o_output)
        
        return self.o_output

       
     #========================End implementation section 1==============================================="   
        
        
        
        
     #========================Begin implementation section 2=============================================#    

    def backPropagate(self, targets):
        
        # calculate error terms for output
        self.errors = self.o_output - targets
        delta_e_u_output = self.errors * self.o_output * (1 - self.o_output)
        delta_e_u_horizontal = np.matrix(delta_e_u_output)
        o_hidden_vertical = np.matrix(self.o_hidden).T
        
        delta_e_w_output = np.dot(o_hidden_vertical, delta_e_u_horizontal)

        # calculate error terms for hidden
        delta_e_u_hidden = np.dot(self.W_hidden_to_output, delta_e_u_output) * self.o_hidden * (1 - self.o_hidden)
        delta_e_u_horizontal = np.matrix(delta_e_u_hidden)
        o_input_vertical = np.matrix(self.a_input).T
        delta_e_w_hidden = np.dot(o_input_vertical, delta_e_u_horizontal)
        # delete last column
        # delta_e_w_hidden = delta_e_w_hidden[:,0:delta_e_w_hidden.shape[1]-1]
        delta_e_w_hidden = np.delete(delta_e_w_hidden, -1, 1)
        # update output weights
        self.W_hidden_to_output -= self.learning_rate * delta_e_w_output
        # update input weights
        self.W_input_to_hidden -= self.learning_rate * delta_e_w_hidden
        
        return np.square(self.errors).sum()/2
        
     #========================End implementation section 2 =================================================="   

    
    
    
    def train(self, data,validation_data):
        start_time = time.time()
        errors=[]
        Training_accuracies=[]
      
        for it in range(self.iterations):
            np.random.shuffle(data)
            inputs  = [entry[0] for entry in data ]
            targets = [ entry[1] for entry in data ]
            error=0.0 
            
            for i in range(len(inputs)):
                Input = inputs[i]
                Target = targets[i]
                self.feedForward(Input)
                error+=self.backPropagate(Target)
            Training_accuracies.append(self.predict(data))
            
            error=error/len(data)
            errors.append(error)
            
           
            print("Iteration: %2d/%2d[==============] -Error: %5.10f  -Training_Accuracy:  %2.2f  -time: %2.2f " %(it+1,self.iterations, error, (self.predict(data)/len(data))*100, time.time() - start_time))
            # you can add test_accuracy and validation accuracy for visualisation 
            
        plot_curve(range(1,self.iterations+1),errors, "Error")
        plot_curve(range(1,self.iterations+1), Training_accuracies, "Training_Accuracy")
       
        
     

    def predict(self, test_data):
        """ Evaluate performance by counting how many examples in test_data are correctly 
            evaluated. """
        count = 0.0
        for testcase in test_data:
            answer = np.argmax( testcase[1] )
            prediction = np.argmax( self.feedForward( testcase[0] ) )
            count = count + 1 if (answer - prediction) == 0 else count 
            count= count 
        return count 
    
    
    
    def save(self, filename):
        """ Save neural network (weights) to a file. """
        with open(filename, 'wb') as f:
            pickle.dump({'wi':self.W_input_to_hidden, 'wo':self.W_hidden_to_output}, f )
        
        
    def load(self, filename):
        """ Load neural network (weights) from a file. """
        with open(filename, 'rb') as f:
            data = pickle.load(f)
        # Set biases and weights
        self.W_input_to_hidden=data['wi']
        self.W_hidden_to_output = data['wo']
        
            
                                  
                                  
    
  



    
    
   