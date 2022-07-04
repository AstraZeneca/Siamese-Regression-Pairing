![Maturity level-0](https://img.shields.io/badge/Maturity%20Level-ML--0-red)

This repository corresponds to the article'Siamese Neural Networks for Regression: Similarity-Based Pairing and Uncertainty Quantification'. The project consists of 4 models: Multilayer perceptron model with single input (MLP-FP) and paired input (MLP-deltaFP), [Chemformer model](https://github.com/MolecularAI/Chemformer)[[1]](#1) , and Siamese model with Chemformer strcuture (Cheformer-snn)


MLP-deltaFP: 

for exaustive pairs:

`python mlp.py -s lipo_all.yml -st 0 -f lipo`  

for similarity-based pairs:

`python mlp.py -s lipo_top1.yml -st 1 -f lipo`  

MLP-FP:

`python mlp.py -s lipo_mlp.yml`

Cheformer-snn:

for dropout 0.0:

`python finetuenRegr_k_fold.py --name lipo --data_path lipo/ --drp 0.0`   

we need to run dropout = [0.0,0.05,0.1,0.17]

Chemformer:

`python finetuneRegr_k_fold.py --name lipo --data_path lipo/` 

generate plots:

`python confidence_plot.py` 

`python dropout_plot.py` 

`python plot_n_shot.py` 

`python shot_plot.py` 



<a id="1">[1]</a>
Irwin, R., Dimitriadis, S., He, J., Bjerrum, E.J., 2021. Chemformer: A Pre-Trained Transformer for Computational Chemistry. Mach. Learn. Sci. Technol. [https://doi.org/10.1088/2632-2153/ac3ffb](https://doi.org/10.1088/2632-2153/ac3ffb)

            
