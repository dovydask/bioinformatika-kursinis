import numpy as np

def sigmoid(x, derivative=False):
	if(derivative==True):
		return x*(1-x)
	return 1/(1+np.exp(-x))
	
input = np.array([ 	[0,0,1],
					[0,1,1],
					[1,0,1],
					[1,1,1]	])
					
output = np.array([	[0],
					[1],
					[1],
					[0]])

np.random.seed(1)

synapse_1 = 2*np.random.random((3,4)) - 1
synapse_2 = 2*np.random.random((4,1)) - 1

for i in range(100000):
	
	inp_layer = input
	layer_1 = sigmoid(np.dot(inp_layer, synapse_1))
	layer_2 = sigmoid(np.dot(layer_1, synapse_2))
	
	layer_2_error = output - layer_2
	
	if(i%10000) == 0:
		print("Error: " + str(np.mean(np.abs(layer_2_error))))
	
	layer_2_delta = layer_2_error*sigmoid(layer_2, derivative=True)
	
	layer_1_error = layer_2_delta.dot(synapse_2.T)
	layer_1_delta = layer_1_error * sigmoid(layer_1, True)
	
	synapse_2 += layer_1.T.dot(layer_2_delta)
	synapse_1 += inp_layer.T.dot(layer_1_delta)
	
print("Output after training: ")
print(layer_2)