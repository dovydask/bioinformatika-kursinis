import numpy as np

# Paprasto sigmoid neuronų tinklo simuliacija. Tinklas treniruotei gauna
# kokį nors skaičių 10-ies elementų bitų masyvų ir vieną tokį patį įvesties masyvą.
# Tinklas pagal gautą treniruotę bando nustatyti, kiek vienetų yra įvestyje.
#
# Autorius: Dovydas Kičiatovas

#Sigmoid funkcija
def sigmoid(x, derivative=False):	
	if(derivative==True):
		return x*(1-x)
	return 1/(1+np.exp(-x))
	
#Treniruotės masyvas
examples = np.array([ 	[1,0,1,1,0,0,1,0,1,0,1,0,0,0,0,0,1,0,1,0,1,0,0,0,0,0,1,0,1,0,1,0,0,0,0,0,1,0,1,0,1,0,0,0,0,0,1,0,1,0],
						[1,1,0,1,0,1,1,0,1,0,1,0,1,0,1,0,0,0,0,0,1,0,0,0,0,0,1,0,1,0,1,0,0,0,0,0,1,0,1,0,1,0,0,0,0,0,1,0,1,0],
						[0,1,1,0,1,1,0,0,1,0,1,0,0,0,0,0,1,0,1,0,1,0,0,0,0,0,1,0,0,1,0,1,0,1,1,0,0,0,0,1,0,1,0,0,0,0,1,0,1,0],
						[1,1,1,1,0,0,1,0,1,0,1,1,0,0,0,0,1,0,1,0,1,0,0,0,0,0,1,0,1,0,1,0,0,0,0,0,1,0,1,0,1,0,0,0,0,0,1,0,1,0],
						[1,1,1,0,1,0,1,0,1,0,0,1,0,1,0,1,0,0,1,1,0,0,1,0,1,0,1,1,0,0,0,0,1,0,1,0,1,1,0,1,1,1,1,1,0,1,0,1,0,1],
						[0,1,0,1,0,1,0,0,1,0,1,0,0,0,0,0,1,0,1,0,1,0,1,0,0,1,0,0,0,0,1,0,1,0,1,0,0,0,1,0,1,0,0,0,0,0,1,0,1,0],
						[1,1,0,1,0,0,1,0,1,0,1,0,1,0,0,0,1,0,1,0,1,0,0,0,0,0,1,0,1,0,1,0,0,0,0,0,1,0,1,0,1,0,0,0,0,0,1,0,1,0],
						[1,1,1,1,0,0,0,1,1,0,1,0,0,0,0,0,1,0,1,0,1,0,0,0,0,0,1,0,1,0,1,0,0,0,0,0,1,0,1,0,1,0,0,0,0,0,1,0,1,0],
						[0,1,1,0,0,1,1,1,1,1,0,0,1,0,1,0,1,1,0,1,0,0,0,0,0,0,1,0,1,0,0,0,1,0,0,0,1,0,1,0,1,0,0,0,0,0,1,0,1,0],
						[1,0,1,0,0,0,0,1,1,0,1,0,0,0,0,0,1,0,1,0,1,0,0,0,0,0,1,0,1,0,1,0,0,0,0,0,1,0,1,0,1,0,0,0,0,0,1,0,1,0]])

#Įvesties masyvas						
input = np.array([	[1],
					[0],
					[0],
					[0],
					[0],
					[1],
					[0],
					[0],
					[1],
					[1]	])

#np.random.seed(1)

#Sinapsės inicializuojamos su atsitiktiniais svoriais
synapse_1 = 2*np.random.random((50,10)) - 1
synapse_2 = 2*np.random.random((10,1)) - 1

#Tinklo treniravimas
for i in range(5000):
	
	inp_layer = examples
	layer_1 = sigmoid(np.dot(inp_layer, synapse_1))
	layer_2 = sigmoid(np.dot(layer_1, synapse_2))
	
	layer_2_error = input - layer_2
	
	#if(i%10000) == 0:
		#print("Tikslumas: " + str(1 - np.mean(np.abs(layer_2_error))))
	
	layer_2_delta = layer_2_error*sigmoid(layer_2, derivative=True)
	
	layer_1_error = layer_2_delta.dot(synapse_2.T)
	layer_1_delta = layer_1_error * sigmoid(layer_1, True)
	
	synapse_2 += layer_1.T.dot(layer_2_delta)
	synapse_1 += inp_layer.T.dot(layer_1_delta)

#Tinklo paklaida nuo įvesties
accuracy = np.mean(np.abs(layer_2_error))
print("Paklaida: " + str(accuracy))
print("Išeitis po treniravimo: ")
print(layer_2)

counter = 0
bitcount = 0

for i in range(10):
	#Siekiant tikslesnio spėjimo paklaidą sumažinau 0.01, nes atėmus tinklo paklaidą
	#tik iš vieneto, gali gautis didesni skaičiai nei tinklo išeities nustatytos reikšmės.
	#Galbūt tiktų koks nors kitas palyginimo metodas?
	if(layer_2[i] > 1-accuracy-0.01):
		counter=counter+1
	
	if(input[i] == 1):
		bitcount=bitcount+1
		
print("Vienetų skaičius (spėjimas): " + str(counter));
print("Vienetų skaičius (tikrasis): " + str(bitcount));
#print("Tikslumas: " + str(counter/bitcount))