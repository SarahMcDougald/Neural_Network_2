
###

"""
# PROJECT GOALS: 
1) Create a neural network in Python, using the Pandas and Numpy libraries, that is capable of learning a basic y=x^2 pattern accurately (over 95%).
2) Create a NN that can predict the price of a house in Beaverton, given two relevant variables, within 10% of the actual value. 

Inspiration for project: tutorial by "Data Boys Learning" at https://databoys.github.io/Feedforward/
What separates this project from the tutorial:
- use of matplotlib.pyplot functions to visualize the housing data
- use of pandas functions to load and prepare data for machine learning processes
- two different applications of the neural network: a linear equation, and house prices
- **a different activation function: LeakyRELU (and its derivative) used here instead of the sigmoid function. 

- FUTURE GOAL: expand to having constructors for four-layers, etc...

"""

###
# Note: all data is from Beaverton houses, using the website Zillow.com. Data was retrieved from the following URL:
# https://www.zillow.com/homes/for_sale/Beaverton-OR/pmf,pf_pt/house,mobile_type/30381_rid/1-_beds/1-_baths/50000-1000000_price/182-3647_mp/pricea_sort/45.563823,-122.675229,45.405199,-122.970486_rect/11_zm/
# 
# Project started 09/03/2017
# 
# There are three strategies for loading data: vanilla Python, NumPy, and Pandas. Source: https://machinelearningmastery.com/load-machine-learning-data-python/

###


import csv
import numpy as np
import pandas
import matplotlib.pyplot as plt


############################################################## PANDAS #####################################################################


# Careful of the file name.
fileName = "D:\PythonProjects\Neural_Network_2\houses_sqft_and_numberOfRooms.csv"
columnNames = ['sqft', 'rooms', 'price']
data = pandas.read_csv(fileName) #data = pandas.read_csv(fileName, names=columnNames)


# TODO: put Pandas loading here. 
# TODO: put matplotlib functions here: one for y=x^2, one for housing data.


#########################################################  NEURAL NETWORK ##################################################################



def leakyRELU(value):
    """
    Activation function that will be used in this NN, the so-called Leaky RELU.

    param value : the numeric double that will be input into the activation function. Will be input*weight...

    """
    if (value >= 0):
        return value
    else:
        # Consider changing this multiplier.
        return .01 * value



def derivRL(value):
    """
    Derivative of the activation function, Leaky RELU. 

    This function will be called twice, in two backpropagation steps:
    - when calculating the error between output layer and hidden layer
    - when calculating the error between hidden layer and input layer

    param value : ** what will be input into this? 
    
    BE CAREFUl.
    """
    if value == 0:
        return 0
    else:
        return 1




# Neural network class definition, with all its "abilities" (member functions).

class Neural_Network(object):
    def __init__(self, numInputs, numHiddens, numOutputs):
        """
        Constructor for Neural_Network class.
        Note that this constructor defaults to a neural network with ONE (1) hidden layer. 

        param numInputs : the number of nodes in the input layer
        param numHiddens : the number of nodes in the hidden layer
        param numOutputs : the number of nodes in the output layer

        """

        # A neural network possesses an input layer, a hidden layer, an output layer, and-- since this is a THREE-layer network-- 3-1 = 2 sets of connections.
        # Declare class variables.

        # Total numbers of nodes in each layer, for keeping track.
        self.inputs_numNodes = numInputs #Optional bias node.
        self.hidden_numNodes = numHiddens
        self.outputs_numNodes = numOutputs

        # TODO: read up on 'bias' nodes. How many do I need? Should there be one in input AND hidden layer? What are they for, and what to set them to? Adjust them?

        # Create a list for each layer, initializing the values to 1.0.
        self.inputs_rawValues = [1.0] * inputs_numNodes
        self.hidden_rawValues = [1.0] * hidden_numNodes
        self.outputs_rawValues = [1.0] * outputs_numNodes


        # TODO: fill in each layer with the proper values.... Or, is the DataFrame for training? Call it later? --Double-check logic.--

        # Create the 2 sets of connections. There needs to be a list of actual weight values for each layer, and a list of weight changes for each layer.
        # Weights are randomized to begin with.
        # Justification for using randn(): ___*fill*___
        self.layer1_weightValues = np.random.randn(self.inputs_numNodes, self.hidden_numNodes)
        self.layer2_weightValues = np.random.randn(self.hidden_numNodes, self.outputs_numNodes)

        self.layer1_weightChanges = np.zeros(self.inputs_numNodes, self.hidden_numNodes)
        self.layer2_weightChanges = np.zeros(self.hidden_numNodes, self.outputs_numNodes)

    def feedForward(self, inputValues):
        """

        Description: Function will fill nodes with raw values, calculate the sum (of each input * connection), put this through an activation function, 
        and do so for each hidden node. Repeat for hidden/output layer connections.

        param inputValues : the values for each input node.

        """
        # FIRST set the input raw values to the inputValues.
        for i in (self.inputs_numNodes):
            self.inputs_rawValues[i] = inputValues[i]

        #result = 0

        # Input and Hidden Layer:
        for hNode in range(self.hidden_numNodes):
            newVal = 0
            for iNode in range(self.inputs_numNodes):
                #Two ways to go about this: make a bucket, add it at the end, or just do it all in one nasty line.
                newVal += self.inputs_rawValues[iNode] * self.layer1_weightValues[iNode][hNode]
            self.hidden_rawValues[hNode] = leakyRELU(newVal)
        
        # Hidden and Output Layer:
        for oNode in range(self.outputs_numNodes):
            newVal = 0
            for hNode in range(self.hidden_numNodes):
                newVal += self.hidden_rawValues[hNode] * self.layer2_weightValues[hNode][oNode]
            self.outputs_rawValues[oNode] = leakyRELU(newVal)
        
        # Now find the "result": basically return whatever the raw values of the output nodes are.
        # ACTUALLY ... since they were altered by this function and are a class variable, you don't have to return them. Just access them in backPropagate().


        #return result


    def backPropagate(self, targets):
        """

        Function to backpropagate.

        Parameters???
        """

        # Calculate error between output layer and hidden layer: 
        # -- find the error for each value in outputs_rawValues[i]
        # -- multiply that individual error by derivLeakyRELU(outputs_rawValues[i])

        for oNode in range(self.outputs_numNodes):
            #




        #Should have no return. Just change the weights.


    def train(self, dataPairs, iterations, learningRate):
        """
        NOTE: May want to move variable 'learningRate' to the constructor for a Neural Network, rather than this function. 
        (So that backPropagate() can use it.)


        A function to train the neural network.
        """
        for cycle in range(iterations):
            for input in range(len(dataPairs[0]) ):
                # For each data pair, run feedforward() and backpropagate(). 
                bucket = self.feedForward(dataPairs[0]) # Need to input ONLY the inputs.. not the whole 2D array of data pairs...
                self.backPropagate(dataPairs[1])


                # NOTE: ____ PROBLEM. This format is not generalizable to neural networks with more than one input node..... and more than one output node...
                

                
                


def test1_xSquared():
    """
    A neural network test on the equation y=x^2. 
    NN is a 3-layer MLP with one input node, two hidden nodes, and three output nodes.

    TODO: fix function calls, after you finish with the training functions' architecture....

    """
    # Create a basic set of data on which to train the network. Assume: 1 input node, 2 hidden nodes, 1 output node. x^2.
    xSquaredPattern = [ [1, 2, 3, 4, 5], [1, 4, 9, 16, 25] ]
    print(xSquaredPattern)
    print(xSquaredPattern[0])

    net1 = Neural_Network(1,2,1)
    net1.train(xSquaredPattern) # Add more variables to this.

    # Note: may not need a built-in predict function? Unsure. Probably will.
    net1.predict(7)

def test2_housingPrices():
    """
    A neural network test on the housing prices in a specific location. 
    NN is a 3-layer MLP with two input nodes, three hidden nodes, and one output node. 

    TODO: Make sure to fix function calls, after you finish with the training functions' architecture....

    """
    #Insert pandas data loading here...

    net2 = Neural_Network(2,3,1)


# Call your test cases.
test1_xSquared()








