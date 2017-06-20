import numpy as np
import matplotlib.pyplot as plt
np.set_printoptions(threshold=np.nan)

# Paprasto 4-ių sluoksnių sigmoid neuronų tinklo simuliacija. 
# 
# Tinklas treniruotei gauna kokį nors skaičių atsitiktinių 
# 8-ių bitų masyvų su atitinkamais taikiniais (9 elementų masyvais). 
# 
# Tinklo testavimui pateikiamas nurodytas skaičius atsitiktinių įvesties 
# masyvų (taip pat 8 elementų). 
# 
# Naudojamas "Mini-batch" gradientas (nustačius mini_batch_size = 1 
# gaunamas stochastinis gradientas).
# 
# Tinklas pagal gautą treniruotę bando nustatyti, kiek vienetų yra įvestyje.
#
# Autorius: Dovydas Kičiatovas

# np.random.seed(1)

# Tinklo nustatymai
example_count = 200			# Treniruotei skirtų pavyzdžių skaičius
input_count = 50			# Testavimui skirtų įvesčių skaičius
mini_batch_size = 1			# Pavyzdžių poaibio dydis mini-batch gradientui
epochs = 5000				# Iteracijų skaičius mokymui
bit_count = 8				# Pavyzdinio masyvo dydis (bitų skaičius)
learning_rate = 1			# Mokymosi greitis
hidden_layer_size = 25		# Paslėptojo sluoksnio dydis

# Sigmoid funkcija
def sigmoid(x, derivative=False):	
	if(derivative==True):
		return x*(1-x)
	return 1/(1+np.exp(-x))

# Treniruotės masyvas (atsitiktinis)
examples = np.zeros((example_count, bit_count), dtype=np.int)	
for i in range(examples.shape[0]):
	for j in range(examples.shape[1]):
		examples[i][j] = np.random.randint(0, 2)
		
# Įvesties masyvas (atsitiktinis)						
input = np.zeros((input_count, bit_count))
for i in range(input.shape[0]):
	for j in range(input.shape[1]):
		input[i][j] = np.random.randint(0, 2)

# Tikrasis įvestyse esančių bitų skaičius (rezultato palyginimui)
input_value = np.zeros(input.shape[0])
for i in range(input.shape[0]):
	counter = 0
	for j in range(input.shape[1]):
		if(input[i][j] == 1):
			counter = counter + 1
	input_value[i] = counter

# Tikroji pavyzdžių reikšmė (pageidaujama išvestis - taikiniai)
output = np.zeros((examples.shape[0], examples.shape[1]+1), dtype=np.int)
for i in range(examples.shape[0]):
	counter = 0
	for j in range(examples.shape[1]):
		if(examples[i][j] == 1):
			counter = counter + 1
	output[i][counter] = 1

# Sinapsės inicializuojamos su atsitiktiniais svoriais
synapse_1 = 2*np.random.random((bit_count, hidden_layer_size)) - 1
synapse_2 = 2*np.random.random((hidden_layer_size, bit_count+1)) - 1

# Pavyzdžių poaibių sudarymas
def iterate_minibatches(inputs, targets, batchsize, shuffle=False):
    assert inputs.shape[0] == targets.shape[0]
    if shuffle:
        indices = np.arange(inputs.shape[0])
        np.random.shuffle(indices)
    for start_idx in range(0, inputs.shape[0] - batchsize + 1, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
        yield inputs[excerpt], targets[excerpt]
	
# Masyvas paklaidoms (grafikui)	
errors = []

# Tinklo treniravimas
for x in range(epochs):
	print(x)
	for batch in iterate_minibatches(examples, output, mini_batch_size, False):
		
		# Paimamas pavyzdžių ir atitinkamų taikinių poaibis
		batch_input, batch_output = batch
		
		layer_0 = batch_input
		output_ref = batch_output
		
		# Pavyzdžių matrica sudauginama su sinapsės matrica ir gaunama
		# sigmoid funkcijos reikšmė kiekvienam elementui
		layer_1 = sigmoid(np.dot(layer_0, synapse_1))
		layer_2 = sigmoid(np.dot(layer_1, synapse_2))
		
		# Gaunamos paklaidos ir gradientas kiekvienam sluoksniui
		layer_2_error = layer_2 - output_ref
		layer_2_delta = layer_2_error*sigmoid(layer_2, True)
				
		layer_1_error = layer_2_delta.dot(synapse_2.T)
		layer_1_delta = layer_1_error*sigmoid(layer_1, True)
		
		# Konvertavimas į Numpy matricas (skaičiavimo paprastumui)
		layer_1 = np.matrix(layer_1)
		layer_2_delta = np.matrix(layer_2_delta)
		layer_2 = np.matrix(layer_2)
		
		# Pagal suskaičiuotus gradientus keičiami svoriai
		synapse_2 -= learning_rate * layer_1.T.dot(layer_2_delta)
		synapse_1 -= learning_rate * layer_0.T.dot(layer_1_delta)
		
	errors.append(np.mean(np.abs(layer_2_error)))

print(np.mean(np.abs(layer_2_error)))	
#Tinklo testavimas
print("Spėjimas / Tikroji reikšmė")
correct_guesses = 0		#Teisingų spėjimų skaičius
ind_taiklumas = []		#Kiekvienos įvesties bitų skaičiaus spėjimo tikslumas
for i in range(input.shape[0]):
	layer_0 = input[i]
	
	layer_1 = sigmoid(np.dot(layer_0, synapse_1))
	layer_2 = sigmoid(np.dot(layer_1, synapse_2))
	
	temp = None
	index1 = 0
	index2 = input_value[i]
	for j in range(input.shape[1]):
		if(temp is None or layer_2[j] > temp):
			temp = layer_2[j]
			index1 = j
		
	print(index1, " / ", int(index2))
	if(index1 == index2):
		if(index2 == 0):
			index2 = 1
		correct_guesses = correct_guesses + 1
		ind_taiklumas.append(index1/index2)
	elif(index1 > index2):
		if(index1 == 0):
			index1 = 1
		ind_taiklumas.append(index2/index1)
	elif(index1 < index2):
		if(index2 == 0):
			index2 = 1
		ind_taiklumas.append(index1/index2)
		
print("Teisingi spėjimai: ", correct_guesses, " iš ", input.shape[0], "(", correct_guesses/input.shape[0], ")")

sum = 0
for i in range(len(ind_taiklumas)):
	sum = sum + ind_taiklumas[i]
	
print("Individualių spėjimų taiklumo vidurkis: ", sum/len(ind_taiklumas))

plt.xlabel('Iteracijos')
plt.ylabel('Paklaida')
plt.plot(np.arange(100), errors[0:100])
plt.show()	