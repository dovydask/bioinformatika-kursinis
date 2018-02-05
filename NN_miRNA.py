import numpy as np					
from Bio import SeqIO
import matplotlib.pyplot as plt
import theano
import theano.tensor as T
import time

np.set_printoptions(threshold=np.nan)
print(theano.__version__)
# Autorius: Dovydas Kičiatovas
# np.random.seed(1)

# Tinklo nustatymai								
prop_80_20 = True 				    # Nurodo, kokias duomenų proporcijas naudoti
								    # imant imtis mokymui ir validavimui. Jei False,
								    # naudojama 90-10 proc. atitinkamai. Jei True, 
								    # naudojama 80-20 proc.;

mini_batch_size = 128		        # Mokymo pavyzdžių poaibis Mini-Batch gradientinio
								    # mokymo tipui;

epochs = 2000					    # Iteracijų skaičius mokymui;
learning_rate = 0.01				# Mokymosi greitis;
frame_size = 100 				    # Rėmo dydis, pagal kurį skirstoma baltymo seka;
hidden_layer_1_size = 128		    # Pirmojo paslėptojo sluoksnio dydis;
hidden_layer_2_size = 64		    # Antrojo paslėptojo sluoksnio dydis;
max_prot_length = 200		        # Įvesties sekos ilgio apribojimas;
example_expansion_step_size = 10 	# Žingsnio dydis, pagal kurį juda sekos rėmas;
cutoff = 0.995
chrom_step_size = frame_size

class_1_input_file = "C:/Users/Dovydas/Desktop/kursinis_2/theano_rnn/Neural Network/miRNAs/pre-miRNAs.fasta"
class_2_input_file = "C:/Users/Dovydas/Desktop/kursinis_2/theano_rnn/Neural Network/miRNAs/random.txt"
test_input_file = "C:/Users/Dovydas/Desktop/kursinis_2/theano_rnn/Neural Network/miRNAs/test.txt"
chrom_input = "C:/Users/Dovydas/Desktop/kursinis_2/theano_rnn/Neural Network/miRNAs/chr21.fa"

class1_target = [1]			# Real pre-miRNA
class2_target = [0]			# Random pre-miRNA
output_size = 1				# Išvesties sluoksnio dydis.

print("*------------------Režimas------------------*")
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

# Nukleotidų skalė konvertavimui į skaičius.
nucleotides = {
    'N' : 0.00, 
    'A' : 0.25,
    'U' : 0.50,
    'G' : 0.75,
    'C' : 1.00
}

num_to_nucl = {
    0.00 : 'N',
    0.25 : 'A',
    0.50 : 'U',
    0.75 : 'G',
    1.00 : 'C'
}

# Sekos nukleotidų konvertavimas į skaičius.
def convert_sequence(sequence):
    split = list(sequence)
    for i in range(0, len(split)):
        split[i] = nucleotides[split[i]]
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
    all_alpha_examples, all_alpha_outputs = read_example_sequences(class_1_input_file, class1_target)
    all_beta_examples, all_beta_outputs = read_example_sequences(class_2_input_file, class2_target)

    #all_alpha_examples = all_alpha_examples+all_alpha_examples+all_alpha_examples+all_alpha_examples+all_alpha_examples+all_alpha_examples+all_alpha_examples+all_alpha_examples+all_alpha_examples+all_alpha_examples
    #all_alpha_outputs = all_alpha_outputs+all_alpha_outputs+all_alpha_outputs+all_alpha_outputs+all_alpha_outputs+all_alpha_outputs+all_alpha_outputs+all_alpha_outputs+all_alpha_outputs+all_alpha_outputs
    
    print("Real-miRNR skaičius (praplėstas)")
    print(len(all_alpha_examples))
    print()
    print("Pseudo-miRNR skaičius (praplėstas)")
    print(len(all_beta_examples))
    print()

    return all_alpha_examples+all_beta_examples, all_alpha_outputs+all_beta_outputs

# Įvesčių tinklo spėjimui testuoti paruošimas.
def prepare_inputs():
	all_inputs, tempout = read_example_sequences(test_input_file, [1])
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

chromosome = SeqIO.parse(open(chrom_input),'fasta')
chrom = ""
for sequence in chromosome:
    chrom = str(sequence.seq)

chrom = chrom.upper()
chrom = chrom.replace('T', 'U')
chromosome = convert_sequence(chrom)
chromosome_length = len(chromosome)

chromosome = theano.shared(value=np.asarray(chromosome, dtype=theano.config.floatX))
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
chromosome_index = T.iscalar()
x = T.fmatrix('x')
y = T.fmatrix('y')
cx = T.fvector('cx')

chromosome_test = T.nnet.sigmoid(T.dot(T.nnet.sigmoid(T.dot(T.nnet.sigmoid(T.dot(cx, synapse_1)), synapse_2)), synapse_3))
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

chrom_test = theano.function(
    inputs=[chromosome_index],
    outputs=[chromosome_test],
    givens={
        cx : chromosome[chromosome_index : chromosome_index + frame_size]
    },
    allow_input_downcast=True
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
	if(validation_results[0][i] >= cutoff and outputs[i] == 1):
		successes += 1
	elif(validation_results[0][i] < cutoff and outputs[i] == 0):
		successes += 1

print("Validation results: ", successes/len(validation_results[0]))
# Tinklo spėjimai iš vartotojo įvesties
test_results = test()
print(test_results)

dens_count = 0
chrom_test_results = []
densities = []
start_time = time.time()

for i in range(0, chromosome_length - frame_size, chrom_step_size):
    if chrom_test(i)[0][0] >= cutoff:
        chrom_test_results.append(1)
        dens_count += 1
        chr_out = chrom[i:i+chrom_step_size]
        #chr_out = chr_out.replace('U', 'T')
        print(chr_out)
    else:
        chrom_test_results.append(0)
    if i % 1000000 == 0:
        dens = dens_count/1000000
        print(i, dens)
        densities.append(dens)
        dens_count = 0

print(time.time() - start_time, " seconds.")

'''
for i in range(0, len(chromosome) - frame_size):
    ch_arr = np.asarray(chromosome[i:i+frame_size], dtype=theano.config.floatX)
    if chrom_test(ch_arr)[0][0] >= 0.5:
        chrom_test_results.append(1)
    else:
        chrom_test_results.append(0)
''' 

# MSE grafiko piešimas.

plt.xlabel('Iteracijos')
plt.ylabel('Paklaida')
plt.plot(np.arange(1, epochs), errors[1:epochs])
plt.show()	

#plt.figure()
#plt.xlabel('i')
#plt.ylabel('y')
#plt.plot(np.arange(1, len(chrom_test_results)), chrom_test_results[1:len(chrom_test_results)], 'ro')
#plt.show()

plt.figure()
plt.xlabel('MBases')
plt.ylabel('Density')
plt.plot(np.arange(1, len(densities)), densities[1:len(densities)])
plt.show()