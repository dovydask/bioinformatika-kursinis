import numpy as np
from Bio import SeqIO

np.set_printoptions(threshold=np.nan)

# Klasės:
# 1 - a/b
# 2 - a+b
# 3 - all beta
# 4 - all alpha
# 5 - multi-domain (alpha and beta)
#
# Autorius: Dovydas Kičiatovas

# np.random.seed(1)

# Tinklo nustatymai
mini_batch_size = 25		# Geriausi (su ~200000 pvz): [20 - 64]
epochs = 100				# Treniruočių (iteracijų) skaičius
learning_rate = 1			# Mokymosi greitis (miu)
frame_size = 350
output_size = 1
hidden_layer_size = 25		# Geriausi: 30, 25
max_prot_length = 500

print()
print("*-------------Tinklo parametrai-------------*")
print("Įvesties sluoksnio dydis: " + str(frame_size))
print("Paslėptojo sluoksnio dydis: " + str(hidden_layer_size))
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
    for i in range(0, len(sequence) - frame_size+1, 50):
        expanded_sequence.append(sequence[i:frame_size+i])
        outputs.append(output)
    return expanded_sequence, outputs
			
def expand_input_sequence(sequence):
    expanded_sequence = []
    while(len(sequence) % frame_size != 0):
        sequence.append(0)		
    for i in range(0, len(sequence) - frame_size+1, 50):
        expanded_sequence.append(sequence[i:frame_size+i])
    return expanded_sequence
			
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
    #print("Įvedamų baltymų skaičius")
    #print(len(sequences))
    #print()
    return sequences, outputs 

def read_input_sequences(input_file):
	fasta_sequences = SeqIO.parse(open(input_file),'fasta')
	input_sequences = {}
	sequences = []
	for fasta in fasta_sequences:
		name, sequence = fasta.id, str(fasta.seq)
		input_sequences[name] = []
		if(len(sequence) <= max_prot_length):
			converted_sequence = convert_sequence(sequence)
			expanded_sequences = expand_input_sequence(converted_sequence)
			input_sequences[name] += expanded_sequences
    #print("Įvedamų baltymų skaičius")
    #print(len(sequences))
    #print()
	return input_sequences
	
def prepare_examples():
    print("*-------------Pavyzdžių kiekiai-------------*")
    all_alpha_examples, all_alpha_outputs = read_example_sequences("SCOP/Complete lists/all-alpha.txt", [0])
    all_beta_examples, all_beta_outputs = read_example_sequences("SCOP/Complete lists/all-beta.txt", [1])
    print("A klasės baltymų skaičius (praplėstas)")
    print(len(all_alpha_examples))
    print()
    print("B klasės baltymų skaičius (praplėstas)")
    print(len(all_beta_examples))
    print()
    return all_alpha_examples+all_beta_examples, all_alpha_outputs+all_beta_outputs
    
def prepare_inputs():
	print("*---------------Testų kiekiai---------------*")
	inputs = read_input_sequences("protein_inputs.txt")
	return inputs
    
expanded_examples, outputs = prepare_examples()
expanded_inputs = prepare_inputs()

print("Visas pavyzdžių kiekis:")
print(len(expanded_examples))
					
print("Visas testinių įvesčių kiekis:")
print(len(expanded_inputs))

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

#expanded_inputs = np.array(expanded_inputs)

for id, input in expanded_inputs.items():
	guesses1 = []
	guesses2 = []
	for i in range(len(input)):
		layer_0 = input[i]
			
		layer_1 = sigmoid(np.dot(layer_0, synapse_1))
		layer_2 = sigmoid(np.dot(layer_1, synapse_2))
			
		if(layer_2 < 0.5):
			guesses1.append(layer_2)
		else:
			guesses2.append(layer_2)
	
	
	if(len(guesses1) >= len(guesses2)):
		print(id.split(':')[0] + " baltymo klasės spėjimas: A (All Alpha)")
	else:
		print(id.split(':')[0] + " baltymo klasės spėjimas: B (All Beta)")
	
#print("Vidurkis: ", np.average(guesses1+guesses2))