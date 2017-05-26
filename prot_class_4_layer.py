import numpy as np
from Bio import SeqIO

np.set_printoptions(threshold=np.nan)

# Autorius: Dovydas Kičiatovas

# np.random.seed(1)

# Tinklo nustatymai
network_4_class_mode = False
prop_80_20 = False 			# False = 90-10

'''
mini_batch_size = 48		# Geriausi (su ~200000 pvz): [20 - 64]
epochs = 100				# Treniruočių (iteracijų) skaičius
learning_rate = 1			# Mokymosi greitis (miu)
frame_size = 500
output_size = 1
hidden_layer_1_size = 48
hidden_layer_2_size = 16
max_prot_length = 1000
example_expansion_step_size = 100
'''

mini_batch_size = 48		# Geriausi (su ~200000 pvz): [20 - 64]
epochs = 100				# Treniruočių (iteracijų) skaičius
learning_rate = 1			# Mokymosi greitis (miu)
frame_size = 500
output_size = 1
hidden_layer_1_size = 48
hidden_layer_2_size = 16
max_prot_length = 1000
example_expansion_step_size = 100

if(network_4_class_mode):
	class1_target = [1,0,0,0]	# All Alpha baltymai
	class2_target = [0,1,0,0]	# All Beta baltymai
	class3_target = [0,0,1,0]	# A+B baltymai
	class4_target = [0,0,0,1]	# A/B baltymai
else:
	class1_target = [0]			# All Alpha baltymai
	class2_target = [1]			# All Beta baltymai

print("*------------------Režimas------------------*")
if(network_4_class_mode):
	print("Tinklo režimas: 4 klasės")
else:
	print("Tinklo režimas: 2 klasės")
if(prop_80_20):
	print("Duomenų proporcijos: 80 proc. pavyzdžių, 20 proc. testams")
	divisor = 5
else:
	print("Duomenų proporcijos: 90 proc. pavyzdžių, 10 proc. testams")
	divisor = 10
print("*-------------Tinklo parametrai-------------*")
print("Įvesties sluoksnio dydis: " + str(frame_size))
print("1-o paslėptojo sluoksnio dydis: " + str(hidden_layer_1_size))
print("2-o paslėptojo sluoksnio dydis: " + str(hidden_layer_2_size))
print("Išeities sluoksnio dydis: " + str(output_size))
print("Iteracijų skaičius (epochs): " + str(epochs))
print("Mokymosi greitis (miu): " + str(learning_rate))
print("Mini-Batch dydis: " + str(mini_batch_size))
print("Maksimalus baltymo ilgis: " + str(max_prot_length))

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

def convert_sequence(sequence):
    split = list(sequence)
    for i in range(0, len(split)):
        split[i] = acids[split[i]]
    return split

def expand_example_sequence(sequence, outputs, output):
    expanded_sequence = []
    while(len(sequence) % frame_size != 0):
        sequence.append(0)		
    for i in range(0, len(sequence) - frame_size+1, example_expansion_step_size):
        expanded_sequence.append(sequence[i:frame_size+i])
        outputs.append(output)
    return expanded_sequence, outputs
			
def read_example_sequences(input_file, output):
    fasta_sequences = SeqIO.parse(open(input_file),'fasta')
    sequences = []
    outputs = []
    for fasta in fasta_sequences:
        sequence = str(fasta.seq)
        if(len(sequence) <= max_prot_length):
            converted_sequence = convert_sequence(sequence)
            expanded_sequences, outputs = expand_example_sequence(converted_sequence, outputs, output)
            sequences += expanded_sequences
    return sequences, outputs 
	
def prepare_examples():
	print("*-------------Pavyzdžių kiekiai-------------*")
	all_alpha_examples, all_alpha_outputs = read_example_sequences("SCOP/Complete lists/all-alpha.txt", class1_target)
	all_beta_examples, all_beta_outputs = read_example_sequences("SCOP/Complete lists/all-beta.txt", class2_target)
	
	print("A klasės baltymų skaičius (praplėstas)")
	print(len(all_alpha_examples))
	print()
	print("B klasės baltymų skaičius (praplėstas)")
	print(len(all_beta_examples))
	print()
	
	if(network_4_class_mode):
		all_plus_examples, all_plus_outputs = read_example_sequences("SCOP/Complete lists/a-plus-b.txt", class3_target)
		all_strich_examples, all_strich_outputs = read_example_sequences("SCOP/Complete lists/a-strich-b.txt", class4_target)
		print("A+B klasės baltymų skaičius (praplėstas)")
		print(len(all_plus_examples))
		print()
		print("A/B klasės baltymų skaičius (praplėstas)")
		print(len(all_strich_examples))
		print()
		return all_alpha_examples+all_beta_examples+all_plus_examples+all_strich_examples, all_alpha_outputs+all_beta_outputs+all_plus_outputs+all_strich_outputs
	else:
		return all_alpha_examples+all_beta_examples, all_alpha_outputs+all_beta_outputs
   
expanded_examples, outputs = prepare_examples()

shuffle_list = list(zip(expanded_examples, outputs))
np.random.shuffle(shuffle_list)
expanded_examples, outputs = zip(*shuffle_list)

expanded_examples = np.array(expanded_examples)
outputs = np.array(outputs)

while(expanded_examples.shape[0] % divisor != 0):
	expanded_examples = expanded_examples[:-1]
	outputs = outputs[:-1]
	
print("Visas duomenų kiekis:")
print(len(expanded_examples))

if(prop_80_20):
	split1, split2, split3, split4, split5 = np.split(expanded_examples, divisor)
	expanded_examples = np.concatenate((split1, split2, split3, split4))
	expanded_inputs = split5
	
	split1, split2, split3, split4, split5 = np.split(outputs, divisor)
	outputs = np.concatenate((split1, split2, split3, split4))
	input_outputs = split5
else:
	split1, split2, split3, split4, split5, split6, split7, split8, split9, split10 = np.split(expanded_examples, divisor)
	expanded_examples = np.concatenate((split1, split2, split3, split4, split5, split6, split7, split8, split9))
	expanded_inputs = split10

	split1, split2, split3, split4, split5, split6, split7, split8, split9, split10 = np.split(outputs, divisor)
	outputs = np.concatenate((split1, split2, split3, split4, split5, split6, split7, split8, split9))
	input_outputs = split10

print("Pavyzdžių kiekis:")
print(len(expanded_examples))
					
print("Testinių įvesčių kiekis:")
print(len(expanded_inputs))

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

#Sinapsės inicializuojamos su atsitiktiniais svoriais
synapse_1 = 2*np.random.random((frame_size, hidden_layer_1_size)) - 1
synapse_2 = 2*np.random.random((hidden_layer_1_size, hidden_layer_2_size)) - 1
synapse_3 = 2*np.random.random((hidden_layer_2_size, output_size)) - 1

print("*------------Tinklo treniravimas------------*")
for x in range(epochs):
	if(x > 0):
		#print(x)
		#print(layer_0.shape)
		#print(layer_1.shape)
		#print(layer_2.shape)
		#print("1 sluoksnio paklaida po " + str(x) + " iteracijos: " + str(np.mean(np.abs(layer_1_error))))
		print("2 ir 3 sluoksnio paklaida po " + str(x) + " iteracijos: " + str(np.mean(np.abs(layer_2_error))) + " " + str(np.mean(np.abs(layer_3_error))))	
	for batch in iterate_minibatches(expanded_examples, outputs, mini_batch_size, False):
		
		batch_input, batch_output = batch
		
		layer_0 = batch_input
		output_ref = batch_output
				
		layer_1 = sigmoid(np.dot(layer_0, synapse_1))
		layer_2 = sigmoid(np.dot(layer_1, synapse_2))
		layer_3 = sigmoid(np.dot(layer_2, synapse_3))
		
		layer_3_error = layer_3 - output_ref
		layer_3_delta = layer_3_error*sigmoid(layer_3, True)
		
		layer_2_error = layer_3_delta.dot(synapse_3.T)
		layer_2_delta = layer_2_error*sigmoid(layer_2, True)
				
		layer_1_error = layer_2_delta.dot(synapse_2.T)
		layer_1_delta = layer_1_error*sigmoid(layer_1, True)
		
		#layer_1 = np.matrix(layer_1)
		#layer_2_delta = np.matrix(layer_2_delta)
		#layer_2 = np.matrix(layer_2)
		
		synapse_3 -= learning_rate * layer_2.T.dot(layer_3_delta)
		synapse_2 -= learning_rate * layer_1.T.dot(layer_2_delta)
		synapse_1 -= learning_rate * layer_0.T.dot(layer_1_delta)

#Tinklo testavimas
successes = []
for i in range(0, expanded_inputs.shape[0]):
	layer_0 = expanded_inputs[i]
		
	layer_1 = sigmoid(np.dot(layer_0, synapse_1))
	layer_2 = sigmoid(np.dot(layer_1, synapse_2))
	layer_3 = sigmoid(np.dot(layer_2, synapse_3))
	
	if(network_4_class_mode and np.argmax(layer_2) == np.argmax(input_outputs[i])):
		successes.append(1)
	elif((input_outputs[i] == 0 and layer_3 < 0.5) or (input_outputs[i] == 1 and layer_3 >= 0.5)):
		successes.append(1)
		
print(len(successes) / expanded_inputs.shape[0])