import numpy as np					
from Bio import SeqIO
import matplotlib.pyplot as plt
import theano
import theano.tensor as T
import time

np.set_printoptions(threshold=np.nan)

# Autorius: Dovydas Kičiatovas
# np.random.seed(1)

# Tinklo nustatymai
network_4_class_mode = False	# Ar naudoti 4 klases klasifikavimui? Jei False,
								# naudojamos 2 klasės.;
								
prop_80_20 = False 				# Nurodo, kokias duomenų proporcijas naudoti
								# imant imtis mokymui ir validavimui. Jei False,
								# naudojama 90-10 proc. atitinkamai. Jei True, 
								# naudojama 80-20 proc.;

mini_batch_size = 64			# Mokymo pavyzdžių poaibis Mini-Batch gradientinio
								# mokymo tipui;
							
epochs = 1000					# Iteracijų skaičius mokymui;
learning_rate = 0.01				# Mokymosi greitis;
frame_size = 300				# Rėmo dydis, pagal kurį skirstoma baltymo seka;
hidden_layer_1_size = 128		# Pirmojo paslėptojo sluoksnio dydis;
hidden_layer_2_size = 64		# Antrojo paslėptojo sluoksnio dydis;
max_prot_length = frame_size			# Baltymo sekos ilgio apribojimas;
example_expansion_step_size = 100	# Žingsnio dydis, pagal kurį juda sekos rėmas;

if(network_4_class_mode):
	class1_target = [1,0,0,0]	# All Alpha baltymai;
	class2_target = [0,1,0,0]	# All Beta baltymai;
	class3_target = [0,0,1,0]	# A+B baltymai;
	class4_target = [0,0,0,1]	# A/B baltymai;
	output_size = 4				# Išvesties sluoksnio dydis;
else:
	class1_target = [0]			# All Alpha baltymai;
	class2_target = [1]			# All Beta baltymai;
	output_size = 1				# Išvesties sluoksnio dydis.

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

# Sigmoid funkcija. Jei derivative = True, grąžinama išvestinės reikšmė.
def sigmoid(x, derivative=False):	
	if(derivative==True):
		return x*(1-x)
	return 1/(1+np.exp(-x))

# Aminorūgščių skalė konvertavimui į skaičius.
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

# Sekos aminorūgščių konvertavimas į skaičius.
def convert_sequence(sequence):
    split = list(sequence)
    for i in range(0, len(split)):
        split[i] = acids[split[i]]
    return split

# Sekos išplėtimas pagal rėmą.
def expand_example_sequence(sequence, outputs, output):
    expanded_sequence = []
    while(len(sequence) % frame_size != 0):
        sequence.append(0)		
    for i in range(0, len(sequence) - frame_size+1, example_expansion_step_size):
        expanded_sequence.append(sequence[i:frame_size+i])
        outputs.append(output)
    return expanded_sequence, outputs
			
# Sekų nuskaitymas iš failo.
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
	
# Pavyzdžių mokymui ir validavimui paruošimas.
def prepare_examples():
	print("*-------------Pavyzdžių kiekiai-------------*")
	all_alpha_examples, all_alpha_outputs = read_example_sequences("C:/Users/Dovydas/Desktop/kursinis_2/theano_rnn/Neural Network/SCOP/all-alpha.txt", class1_target)
	all_beta_examples, all_beta_outputs = read_example_sequences("C:/Users/Dovydas/Desktop/kursinis_2/theano_rnn/Neural Network/SCOP/all-beta.txt", class2_target)
	
	print("A klasės baltymų skaičius (praplėstas)")
	print(len(all_alpha_examples))
	print()
	print("B klasės baltymų skaičius (praplėstas)")
	print(len(all_beta_examples))
	print()
	
	if(network_4_class_mode):
		all_plus_examples, all_plus_outputs = read_example_sequences("C:/Users/Dovydas/Desktop/kursinis_2/theano_rnn/Neural Network/SCOP/a-plus-b.txt", class3_target)
		all_strich_examples, all_strich_outputs = read_example_sequences("C:/Users/Dovydas/Desktop/kursinis_2/theano_rnn/Neural Network/SCOP/a-strich-b.txt", class4_target)
		print("A+B klasės baltymų skaičius (praplėstas)")
		print(len(all_plus_examples))
		print()
		print("A/B klasės baltymų skaičius (praplėstas)")
		print(len(all_strich_examples))
		print()
		return all_alpha_examples+all_beta_examples+all_plus_examples+all_strich_examples, all_alpha_outputs+all_beta_outputs+all_plus_outputs+all_strich_outputs
	else:
		return all_alpha_examples+all_beta_examples, all_alpha_outputs+all_beta_outputs
  
  
# Įvesčių tinklo spėjimui testuoti paruošimas.
def prepare_inputs():
	all_inputs, tempout = read_example_sequences("C:/Users/Dovydas/Desktop/kursinis_2/theano_rnn/Neural Network/protein_inputs.txt", [1])
	return all_inputs
	
expanded_examples, outputs = prepare_examples()

# Pavyzdžiai apjungiami į vieną masyvą ir atsitiktinai sumaišomi.
# Kartu (tuo pačiu išmaišymu) sumaišomas ir atitinkamų taikinių masyvas.
shuffle_list = list(zip(expanded_examples, outputs))
np.random.shuffle(shuffle_list)
expanded_examples, outputs = zip(*shuffle_list)

expanded_examples = np.array(expanded_examples)
outputs = np.array(outputs)
protein_inputs = prepare_inputs()

# Jei pavyzdžių masyvo ilgis nelyginis, pašalinamas paskutinis elementas.
while(expanded_examples.shape[0] % divisor != 0):
	expanded_examples = expanded_examples[:-1]
	outputs = outputs[:-1]
	
print("Visas duomenų kiekis:")
print(len(expanded_examples))

# Pavyzdžių padalijimas į mokymo ir validavimo imtis.
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

# Sinapsės inicializuojamos su atsitiktiniais svoriais

def layer(n_in, n_out):
    return theano.shared(value=np.asarray(rng.uniform(low=-1.0, high=1.0, size=(n_in, n_out)), dtype=theano.config.floatX), name='W', borrow=True)

rng = np.random.RandomState(1234)

synapse_1 = layer(frame_size, hidden_layer_1_size)
synapse_2 = layer(hidden_layer_1_size, hidden_layer_2_size)
synapse_3 = layer(hidden_layer_2_size, output_size)

def shared_dataset(inputs, outputs, tests):
    shared_X = theano.shared(value=np.asarray(inputs, dtype=theano.config.floatX))
    shared_Y = theano.shared(value=np.asarray(outputs, dtype=theano.config.floatX))
    shared_t = theano.shared(value=np.asarray(tests, dtype=theano.config.floatX))
    return shared_X, shared_Y, shared_t

data_x, data_y, data_t = shared_dataset(expanded_examples, outputs, protein_inputs)

index = T.iscalar()
indexv = T.iscalar()
x = T.fmatrix('x')
y = T.fmatrix('y')

validate_output = T.nnet.sigmoid(T.dot(T.nnet.sigmoid(T.dot(T.nnet.sigmoid(T.dot(data_x, synapse_1)), synapse_2)), synapse_3))
test_output = T.nnet.sigmoid(T.dot(T.nnet.sigmoid(T.dot(T.nnet.sigmoid(T.dot(data_t, synapse_1)), synapse_2)), synapse_3))
output = T.nnet.sigmoid(T.dot(T.nnet.sigmoid(T.dot(T.nnet.sigmoid(T.dot(x, synapse_1)), synapse_2)), synapse_3))
err = (y - output)**2
cost = T.sum(err)

train = theano.function(
    inputs=[index], 
    outputs=[err], 
    updates=[
		(synapse_1, synapse_1 - learning_rate*T.grad(cost, synapse_1)),
        (synapse_2, synapse_2 - learning_rate*T.grad(cost, synapse_2)),
        (synapse_3, synapse_3 - learning_rate*T.grad(cost, synapse_3))
    ],
    givens={
        x: data_x[index * mini_batch_size : (index + 1) * mini_batch_size],
        y: data_y[index * mini_batch_size : (index + 1) * mini_batch_size]
    }
)

test = theano.function(
    inputs=[],
    outputs=[test_output]
)

validate = theano.function(
	inputs=[],
	outputs=[validate_output]
)
# Masyvas kiekvienos iteracijos paklaidai talpinti (reikia skaičiuojant MSE).
errors = []
errors.append(0)

# Tinklo mokymas
minibatch_count = int(expanded_examples.shape[0]/mini_batch_size)
assert expanded_examples.shape[0] >= minibatch_count*mini_batch_size
start_time = time.time()
for i in range(epochs):
    print(i, errors[i])
    for minibatch_index in range(minibatch_count):
        error = train(minibatch_index)
        errors.append(np.mean(error))

print(time.time() - start_time, " seconds.")

validation_results = validate()
successes = 0
for i in range(len(validation_results[0])):
	if(validation_results[0][i] >= 0.5 and outputs[i] == 1):
		successes += 1
	elif(validation_results[0][i] <= 0.5 and outputs[i] == 0):
		successes += 1

print("Validation results: ", successes/len(validation_results[0]))
# Tinklo spėjimai iš vartotojo įvesties
test_results = test()
print(test_results)

# MSE grafiko piešimas.
plt.xlabel('Iteracijos')
plt.ylabel('Paklaida')
plt.plot(np.arange(epochs), errors[0:epochs])
plt.show()	