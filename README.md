# bioinformatika-kursinis

2017 m. bioinformatikos kurso kursinio darbo projektas: Baltymų struktūros nustatymas naudojant neuroninį tinklą (Protein structure prediction with a neural network).

Autorius: Dovydas Kičiatovas

Visos programos parašytos Python 3.6.0 programavimo kalba.

Programoms reikalingi Python paketai:
1) Numpy
2) BioPython
3) Matplotlib
4) Theano (tik su grafiniu procesoriumi (GPU) optimizuotiems neuronų tinklams.)

Programoms argumentų nereikia (įsitikinkite, kad failų išdėstymas ir jų vardai tokie patys, kaip repozitorijoje).

Su GPU optimizuoti neuronų tinklai - NN_proteins.py ir NN_miRNA.py

Optimizuotiems tinklams reikalingas grafinis procesorius, palaikantis CUDA (versija 8.0) ir cuDNN (versija 5.1)
