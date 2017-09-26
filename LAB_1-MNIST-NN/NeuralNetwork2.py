import time
import random
import numpy as np
from utils import *
from transfer_functions import * 


class NeuralNetwork2(object):
    
    def __init__(self, input_layer_size, hidden_layer1_size, hidden_layer2_size, output_layer_size, iterations=50, learning_rate = 0.1, transfer='sigmoid'):
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
        self.hidden1 = hidden_layer1_size+1 #+1 for the bias node in the hidden layer 
        self.hidden2 = hidden_layer2_size+1
        self.output = output_layer_size

        # set up array of 1s for activations
        self.a_input = np.ones(self.input)
        self.a_hidden1 = np.ones(self.hidden1)
        self.a_hidden2 = np.ones(self.hidden2)
        self.a_out = np.ones(self.output)
        
        # set up array of 1s for outputs
        self.o_hidden1 = np.ones(self.hidden1)
        self.o_hidden2 = np.ones(self.hidden2)
        self.o_out = np.ones(self.output)
        
        #create randomized weights Yann Lecun method in 1988's paper ( Default values)
        input_range = 1.0 / self.input ** (1/2)
        self.W_input_to_hidden = np.random.normal(loc = 0, scale = input_range, size =(self.input, self.hidden1-1))
        self.W_hidden_to_hidden = np.random.uniform(size = (self.hidden1, self.hidden2-1)) / np.sqrt(self.hidden1)
        self.W_hidden_to_output = np.random.uniform(size = (self.hidden2, self.output)) / np.sqrt(self.hidden2)
        
        # set up array containing the error
        self.errors = np.ones(self.output)
        
        self.transfer_function = sigmoid
        self.dtransfer_function = dsigmoid
        if transfer == 'tanh':
            self.transfer_function = tanh
            self.dtransfer_function = dtanh
        
    def weights_initialisation(self,wi,wh,wo):
        self.W_input_to_hidden=wi # weights between input and hidden layers
        self.W_hidden_to_hidden=wh
        self.W_hidden_to_output=wo # weights between hidden and output layers
   

       
        
    #========================Begin implementation section 1============================================="    
    
    def feedForward(self, inputs):
        self.a_input = np.append(inputs, 1)
        a_hidden_without_bias = np.dot(self.a_input, self.W_input_to_hidden)
        self.a_hidden1 = np.append(a_hidden_without_bias, 0)
        self.o_hidden1 = self.transfer_function(self.a_hidden1)
        self.o_hidden1[-1] = 1
        
        self.a_hidden2 = np.dot(self.o_hidden1, self.W_hidden_to_hidden)
        self.a_hidden2 = np.append(self.a_hidden2, 0)
        self.o_hidden2 = self.transfer_function(self.a_hidden2)
        self.o_hidden2[-1] = 1
        
        self.a_output = np.dot(self.o_hidden2, self.W_hidden_to_output)
        self.o_output = self.transfer_function(self.a_output)
        return self.o_output

       
     #========================End implementation section 1==============================================="   
        
        
        
        
     #========================Begin implementation section 2=============================================#    

    def backPropagate(self, targets):
        
        # calculate error terms for output
        self.errors = self.o_output - targets
        delta_e_u_output = self.errors * self.dtransfer_function(self.o_output)
        delta_e_u_horizontal = np.matrix(delta_e_u_output)
        o_hidden_vertical = np.matrix(self.o_hidden2).T
        delta_e_w_output = np.dot(o_hidden_vertical, delta_e_u_horizontal)
        
        # calculate error terms for hidden 2
        delta_e_u_hidden2 = np.dot(self.W_hidden_to_output, delta_e_u_output) * self.dtransfer_function(self.o_hidden2)
        # delete last column
        delta_e_u_hidden2 = delta_e_u_hidden2[:-1]
        delta_e_u_horizontal2 = np.matrix(delta_e_u_hidden2)
        o_hidden_vertical2 = np.matrix(self.o_hidden1).T
        delta_e_w_hidden2 = np.dot(o_hidden_vertical2, delta_e_u_horizontal2)
        
        # calculate error terms for hidden 1
        delta_e_u_hidden = np.dot(self.W_hidden_to_hidden, delta_e_u_hidden2) * self.dtransfer_function(self.o_hidden1)
        delta_e_u_hidden = delta_e_u_hidden[:-1]
        delta_e_u_horizontal = np.matrix(delta_e_u_hidden)
        o_input_vertical = np.matrix(self.a_input).T
        delta_e_w_hidden = np.dot(o_input_vertical, delta_e_u_horizontal)
        
        # update hidden_output weights
        self.W_hidden_to_output -= self.learning_rate * delta_e_w_output
        # update hidden_hidden weights
        self.W_hidden_to_hidden -= self.learning_rate * delta_e_w_hidden2
        # update input weights
        self.W_input_to_hidden -= self.learning_rate * delta_e_w_hidden
        
        return np.square(self.errors).sum()/2
        
     #========================End implementation section 2 =================================================="   

    
    
    
    def train(self, data,validation_data):
        start_time = time.time()
        errors=[]
        Training_accuracies=[]
        Val_accuracies=[]
      
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
            Val_accuracies.append(self.predict(validation_data))
            
            error=error/len(data)
            errors.append(error)
            
           
            print("Iteration: %2d/%2d[==============] -Error: %5.10f  -Training_Accuracy:  %2.2f  -time: %2.2f " %(it+1,self.iterations, error, (self.predict(data)/len(data))*100, time.time() - start_time))
            # you can add test_accuracy and validation accuracy for visualisation 
            
        plot_curve(range(1,self.iterations+1),errors, "Error")
        plot_curve(range(1,self.iterations+1), Training_accuracies, "Training_Accuracy")
       
        return Val_accuracies
     

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
        
            
                                  
                                  
    
  



    
    
   