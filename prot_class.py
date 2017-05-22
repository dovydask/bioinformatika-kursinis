import numpy as np

np.set_printoptions(threshold=np.nan)

# Klasės:
# 1 - a/b
# 2 - a+b
# 3 - all beta
# 4 - all alpha
# 5 - multi-domain (alpha and beta)
#
#
#
#
#
#
# Autorius: Dovydas Kičiatovas

# np.random.seed(1)

#Neblogas rezultatas (Paklaida sumažinta nuo 0.48 iki 0.35)
#
#mini_batch_size = 32
#epochs = 50
#learning_rate = 1
#frame_size = 200
#output_size = 1
#hidden_layer_size = 25

# Tinklo nustatymai
mini_batch_size = 25		# Geriausi (su ~200000 pvz): [20 - 64]
epochs = 100				# Treniruočių (iteracijų) skaičius
learning_rate = 1			# Mokymosi greitis (miu)
frame_size = 250
output_size = 1
hidden_layer_size = 25		# Geriausi: 30, 25
minimum_prot_length = 250


print()
print("*-------------Tinklo parametrai-------------*")
print("Įvesties sluoksnio dydis: " + str(frame_size))
print("Paslėptojo sluoksnio dydis: " + str(hidden_layer_size))
print("Išeities sluoksnio dydis: " + str(output_size))
print("Iteracijų skaičius (epochs): " + str(epochs))
print("Mokymosi greitis (miu): " + str(learning_rate))
print("Mini-Batch dydis: " + str(mini_batch_size))
print("*-------------Pavyzdžių kiekiai-------------*")
#Sigmoid funkcija
def sigmoid(x, derivative=False):	
	if(derivative==True):
		return x*(1-x)
	return 1/(1+np.exp(-x))

acids = {	'X' : 0.000,
			'R' : 0.050,
			'K' : 0.100,
			'O' : 0.125,
			'N' : 0.150,
			'B' : 0.175,
			'D' : 0.200,
			'E' : 0.250,
			'Z' : 0.275,
			'Q' : 0.300,
			'H' : 0.350,
			'P' : 0.400,
			'Y' : 0.450,
			'W' : 0.500,
			'S' : 0.550,
			'T' : 0.600,
			'G' : 0.650,
			'A' : 0.700,
			'M' : 0.750,
			'C' : 0.800,
			'U' : 0.825,
			'F' : 0.850,
			'L' : 0.900,
			'J' : 0.925,
			'V' : 0.950,
			'I' : 1.000	}

#Treniruotės masyvas (Iš failo, eilutėmis)


def prepare_data_1():
	with open("SCOP/a-strich-b.txt") as file:
		a_strich_b_lines = file.read().splitlines()
	
	a_strich_b_lines.pop(0)
	
	a_strich_b_protein = ""
	a_strich_b_examples = []
	for line in a_strich_b_lines:
		line.rstrip()
		line.lstrip()
		if(line.split()[0][0] == ">"):
			a_strich_b_examples.append(a_strich_b_protein)
			a_strich_b_protein = ""
		elif(line == a_strich_b_lines[len(a_strich_b_lines)-1]):
			a_strich_b_protein += line
			a_strich_b_examples.append(a_strich_b_protein)
			a_strich_b_protein = ""
		else:
			a_strich_b_protein += line
		
	a_strich_b_converted_examples = []
	counter = 0
	for prot in a_strich_b_examples:
		if(prot != ""):
			splitted = list(prot)
			a_strich_b_converted_examples.append([])
			for i in range(0, len(splitted)):
				a_strich_b_converted_examples[counter].append(acids[splitted[i]])
			counter += 1

	a_strich_b_outputs = []
	a_strich_b_expanded_examples = []
	for i in range(0, len(a_strich_b_converted_examples)):
		ex = a_strich_b_converted_examples[i]
		
		while(len(ex) % frame_size != 0):
			ex.append(0)
			
		for j in range(0, len(ex) - frame_size + 1):
			#print(j)
			a_strich_b_expanded_examples.append(ex[j:frame_size+j])
			a_strich_b_outputs.append([1.00, 0, 0, 0])
			
	print("A/B klasės baltymų skaičius (praplėstas)")
	print(len(a_strich_b_expanded_examples))
	print(len(a_strich_b_outputs))		
	
	return a_strich_b_expanded_examples, a_strich_b_outputs

def	prepare_data_2(): 
	with open("SCOP/a-plus-b.txt") as file:
		a_plus_b_lines = file.read().splitlines()
		
	a_plus_b_lines.pop(0)

	a_plus_b_protein = ""
	a_plus_b_examples = []
	for line in a_plus_b_lines:
		line.rstrip()
		line.lstrip()
		if(line.split()[0][0] == ">"):
			a_plus_b_examples.append(a_plus_b_protein)
			a_plus_b_protein = ""
		elif(line == a_plus_b_lines[len(a_plus_b_lines)-1]):
			a_plus_b_protein += line
			a_plus_b_examples.append(a_plus_b_protein)
			a_plus_b_protein = ""
		else:
			a_plus_b_protein += line

			
	a_plus_b_converted_examples = []
	counter = 0
	for prot in a_plus_b_examples:
		if(prot != ""):
			splitted = list(prot)
			a_plus_b_converted_examples.append([])
			for i in range(0, len(splitted)):
				a_plus_b_converted_examples[counter].append(acids[splitted[i]])
			counter += 1

	a_plus_b_outputs = []
	a_plus_b_expanded_examples = []
	for i in range(0, len(a_plus_b_converted_examples)):
		ex = a_plus_b_converted_examples[i]
		
		while(len(ex) % frame_size != 0):
			ex.append(0)
			
		for j in range(0, len(ex) - frame_size + 1):
			#print(j)
			a_plus_b_expanded_examples.append(ex[j:frame_size+j])
			a_plus_b_outputs.append([0, 1.00, 0, 0])
			

	print("A+B klasės baltymų skaičius (praplėstas)")
	print(len(a_plus_b_expanded_examples))
	print(len(a_plus_b_outputs))			
	
	return a_plus_b_expanded_examples, a_plus_b_outputs
	
def prepare_data_3():
	with open("SCOP/Complete lists/all-alpha.txt") as file:
		all_alpha_lines = file.read().splitlines()

	all_alpha_lines.pop(0)
		
	all_alpha_protein = ""
	all_alpha_examples = []
	for line in all_alpha_lines:
		line.rstrip()
		line.lstrip()
		if(line.split()[0][0] == ">"):
			all_alpha_examples.append(all_alpha_protein)
			all_alpha_protein = ""
		elif(line == all_alpha_lines[len(all_alpha_lines)-1]):
			all_alpha_protein += line
			all_alpha_examples.append(all_alpha_protein)
			all_alpha_protein = ""
		else:
			all_alpha_protein += line

	print(len(all_alpha_examples))
	
	all_alpha_converted_examples = []
	counter = 0
	for prot in all_alpha_examples:
		if(prot != "" and len(prot) <= minimum_prot_length):
			splitted = list(prot)
			all_alpha_converted_examples.append([])
			for i in range(0, len(splitted)):
				all_alpha_converted_examples[counter].append(acids[splitted[i]])
			counter += 1

	all_alpha_outputs = []
	all_alpha_expanded_examples = []
	for i in range(0, len(all_alpha_converted_examples)):
		ex = all_alpha_converted_examples[i]
		
		while(len(ex) % frame_size != 0):
			ex.append(0)
			
		for j in range(0, len(ex) - frame_size + 1):
			#print(j)
			all_alpha_expanded_examples.append(ex[j:frame_size+j])
			all_alpha_outputs.append([0])

	print("A klasės baltymų skaičius (praplėstas)")
	print(len(all_alpha_expanded_examples))
	#print(len(all_alpha_outputs))
	print()
	
	return all_alpha_expanded_examples, all_alpha_outputs
	
def prepare_data_4():
	with open("SCOP/Complete lists/all-beta.txt") as file:
		all_beta_lines = file.read().splitlines()

	all_beta_lines.pop(0)
	
	all_beta_protein = ""
	all_beta_examples = []
	for line in all_beta_lines:
		line.rstrip()
		line.lstrip()
		if(line.split()[0][0] == ">"):
			all_beta_examples.append(all_beta_protein)
			all_beta_protein = ""
		elif(line == all_beta_lines[len(all_beta_lines)-1]):
			all_beta_protein += line
			all_beta_examples.append(all_beta_protein)
			all_beta_protein = ""
		else:
			all_beta_protein += line

	print(len(all_beta_examples))
			
	all_beta_converted_examples = []
	counter = 0
	for prot in all_beta_examples:
		if(prot != "" and len(prot) <= minimum_prot_length):
			splitted = list(prot)
			all_beta_converted_examples.append([])
			for i in range(0, len(splitted)):
				all_beta_converted_examples[counter].append(acids[splitted[i]])
			counter += 1

	all_beta_outputs = []
	all_beta_expanded_examples = []
	for i in range(0, len(all_beta_converted_examples)):
		ex = all_beta_converted_examples[i]
		
		while(len(ex) % frame_size != 0):
			ex.append(0)
			
		for j in range(0, len(ex) - frame_size + 1):
			#print(j)
			all_beta_expanded_examples.append(ex[j:frame_size+j])
			all_beta_outputs.append([1.000])
					

	print("B klasės baltymų skaičius (praplėstas)")
	print(len(all_beta_expanded_examples))
	#print(len(all_beta_outputs))
	print()
				
	return all_beta_expanded_examples, all_beta_outputs
	
def prepare_data():
	#print("Pavyzdžiai ruošiami...")
	#expanded_examples1, outputs1 = prepare_data_1()
	#expanded_examples2, outputs2 = prepare_data_2()
	expanded_examples3, outputs3 = prepare_data_3()
	expanded_examples4, outputs4 = prepare_data_4()

	#expanded_examples = expanded_examples1 + expanded_examples2 + expanded_examples3 + expanded_examples4
	#outputs = outputs1 + outputs2 + outputs3 + outputs4
	expanded_examples = expanded_examples3 + expanded_examples4
	outputs = outputs3 + outputs4
	
	return expanded_examples, outputs

	
expanded_examples, outputs = prepare_data()

print("Visas pavyzdžių kiekis:")
print(len(expanded_examples))
#print(len(outputs))
#shuffle_list = list(zip(expanded_examples, outputs))
#np.random.shuffle(shuffle_list)
#expanded_examples, outputs = zip(*shuffle_list)

#print(expanded_examples[0])
#print(outputs[0])

#Įvesties masyvas ()						
with open("protein_inputs.txt") as file:
	lines = file.read().splitlines()

lines.pop(0)
protein = ""
inputs = []
input_count = 1;

for line in lines:
	line.rstrip()
	if(line.split()[0][0] == ">"):
		inputs.append(protein)
		protein = ""
		input_count+=1;
	elif(line == lines[len(lines)-1]):
		protein += line
		inputs.append(protein)
	else:
		protein += line

print(len(inputs))
converted_inputs = []
counter = 0
for prot in inputs:
	splitted = list(prot)
	converted_inputs.append([])
	for i in range(0, len(splitted)):
		converted_inputs[counter].append(acids[splitted[i]])
	counter += 1

print(len(converted_inputs))
expanded_inputs = []

for inp in converted_inputs:
	while(len(inp) % frame_size != 0):
		inp.append(0)
	expanded_inputs.append(inp)
	#for i in range(0, len(inp) - frame_size + 1):
	#	expanded_inputs.append(inp[i:frame_size+i])
			
print(len(expanded_inputs))
'''
for inp in expanded_inputs:
	print(inp)
'''
#print(len(expanded_examples))
#Tikrasis įvestyse esančių bitų skaičius (rezultato palyginimui)

#Tikroji pavyzdžių reikšmė (pageidaujama išvestis - taikiniai iš failo protein_classes.txt)
'''
with open("protein_classes.txt") as file:
	prot_classes = file.read().splitlines()

outputs = []
counter = 0
for prot_class in prot_classes:
	outputs.append([0, 0, 0, 0, 0])
	outputs[counter][int(prot_class)-1] = 1.00
	counter += 1
#print(outputs)
'''
#Sinapsės inicializuojamos su atsitiktiniais svoriais

synapse_1 = 2*np.random.random((frame_size, hidden_layer_size)) - 1
synapse_2 = 2*np.random.random((hidden_layer_size, output_size)) - 1

#synapse_1.fill(0.5)
#synapse_2.fill(0.5)

#Pavyzdžių poaibių sudarymas
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

		
#Tinklo treniravimas
#print(expanded_examples[0])

#expanded_examples = np.array(expanded_examples)
#outputs = np.array(outputs)

shuffle_list = list(zip(expanded_examples, outputs))
np.random.shuffle(shuffle_list)
expanded_examples, outputs = zip(*shuffle_list)

#print(expanded_examples[0])

expanded_examples = np.array(expanded_examples)
outputs = np.array(outputs)

#print(expanded_examples[0])
#print(outputs[0])
print("*------------Tinklo treniravimas------------*")
for x in range(epochs):
	#print(x)
	if(x > 0):
		#print(layer_0)
		#print(layer_1)
		#print(layer_2)
		#print("1 sluoksnio paklaida po " + str(x) + " iteracijos: " + str(np.mean(np.abs(layer_1_error))))
		print("2 sluoksnio paklaida po " + str(x) + " iteracijos: " + str(np.mean(np.abs(layer_2_error))))
	for batch in iterate_minibatches(expanded_examples, outputs, mini_batch_size, False):
		
		batch_input, batch_output = batch
		
		layer_0 = batch_input
		output_ref = batch_output
				
		layer_1 = sigmoid(np.dot(layer_0, synapse_1))
		layer_2 = sigmoid(np.dot(layer_1, synapse_2))
				
		layer_2_error = layer_2 - output_ref
		layer_2_delta = layer_2_error*sigmoid(layer_2, True)
				
		layer_1_error = layer_2_delta.dot(synapse_2.T)
		layer_1_delta = layer_1_error*sigmoid(layer_1, True)
		
		layer_1 = np.matrix(layer_1)
		layer_2_delta = np.matrix(layer_2_delta)
		layer_2 = np.matrix(layer_2)
				
		synapse_2 -= learning_rate * layer_1.T.dot(layer_2_delta)
		synapse_1 -= learning_rate * layer_0.T.dot(layer_1_delta)
		
		#if(i%10000) == 0:
			#print("Tikslumas: " + str(1 - np.mean(np.abs(layer_2_error))))
			
		

#print(layer_2)		
#Tinklo testavimas

expanded_inputs = np.array(expanded_inputs)

guesses1 = []		#Teisingų spėjimų skaičius
guesses2 = []
ind_taiklumas = []		#Kiekvienos įvesties bitų skaičiaus spėjimo tikslumas

cntr = 1
for i in range(expanded_inputs.shape[0]):
	layer_0 = expanded_inputs[i]
		
	layer_1 = sigmoid(np.dot(layer_0, synapse_1))
	layer_2 = sigmoid(np.dot(layer_1, synapse_2))
		
	#print(layer_2)
	if(layer_2 < 0.5):
		print(str(cntr) + " spėjimas: A (All Alpha)")
	else:
		print(str(cntr) + " spėjimas: B (All Beta)")
		
	cntr+=1

#print("Vidurkis: ", np.average(guesses1+guesses2))
'''
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
'''
