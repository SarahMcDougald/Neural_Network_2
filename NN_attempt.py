
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


############################################################## PANDAS #################################################


# Careful of the file name.
fileName = "D:\PythonProjects\Neural_Network_2_Folder\houses_sqft_and_numberOfRooms.csv"
columnNames = ['sqft', 'rooms', 'price']
data = pandas.read_csv(fileName) #data = pandas.read_csv(fileName, names=columnNames)




# Your Pandas toolkit... https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.html


#print(type(data)) # returns <class 'pandas.core.frame.DataFrame'> because data is a DataFrame as defined in pandas.
#print(data.shape) # returns a tuple of the dimensions. (10,3)

#print(data.axes) # returns details of the row and column indices. Pandas fills in on the 'row' with integers. (You can see this with 'print(data)'.)
#print(data.dtypes) # returns the data type of each column.
print("Is dataframe empty: " + str(data.empty))
#print(data.ndim) # Returns 2, the number of dimensions in this data frame... (rows, columns. Nothing else.)
#print(data.notnull()) # Prints a True or False in place of every single element in the array, checking whether or not each element is NULL.

# Print out only one column of a DataFrame. This prints out the 'price' column next to the pandas-assigned index.
print(data.get('price'))

# TODO - AWESOME!!! Makes a not very helpful plot.
#data.plot(x=data['rooms'], y=data['price'], title = "Beaverton Houses by price and rooms")
plt.show()







####################################################  NEURAL NETWORK ##################################################################


# Create a basic set of data on which to train the network. Assume: 1 input node, 2 hidden nodes, 1 output node. x^2.

xSquaredPattern = [ [1, 2, 3, 4, 5], [1, 4, 9, 16, 25] ]
print(xSquaredPattern)
print(xSquaredPattern[0])



def leakyRELU(value):
    """
    Activation function that will be used in this NN, the so-called Leaky RELU.

    param value : the numeric double that will be input into the activation function. Will be input*weight...

    """



def derivRL(value):
    """
    Derivative of the activation function, Leaky RELU. Will be used in backpropagation steps. 

    param value : the double that will be input; input*weight? or backwards equivalent.. 
    """



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
        self.inputs_numNodes = numInputs + 1 # for a "bias node"
        self.hidden_numNodes = numHiddens
        self.outputs_numNodes = numOutputs

        # TODO: read up on 'bias' nodes. How many do I need? Should there be one in input AND hidden layer? What are they for, and what to set them to? Adjust them?

        # Create a list for each layer, initializing the values to 1.0.
        self.inputs_rawValues = [1.0] * inputs_numNodes
        self.hidden_rawValues = [1.0] * hidden_numNodes
        self.outputs_rawValues = [1.0] * outputs_numNodes


        # TODO: fill in each layer with the proper values.... Or, is the DataFrame for training? Call it later? --Double-check logic.--

        # Create the 2 sets of connections. There needs to be a list of actual weight values for each layer, and a list of weight changes for each layer.
        self.layer1_weightValues = np.random.randn(self.inputs_numNodes, self.hidden_numNodes)
        self.layer2_weightValues = np.random.randn(self.hidden_numNodes, self.outputs_numNodes)

        self.layer1_weightChanges = np.zeros(self.inputs_numNodes, self.hidden_numNodes)
        self.layer2_weightChanges = np.zeros(self.hidden_numNodes, self.outputs_numNodes)

    def feedForward(self, inputValues):
        """

        param inputValues : ??? what the hell is this?
        """









