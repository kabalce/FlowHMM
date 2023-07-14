# flow -- cnf -- examples 
Trainin CNF flow and sampling from them. Examples

# Training flow
```bash
python flow_cnf_train.py --example 1 --n_train 3000 --nr_epochs 500 --lrate 0.01 --dims 256 --output-model data_models/flow_model_1dA.pt --output-train-data data_models/train_data_1dA.pkl
python flow_cnf_train.py --example 2 --n_train 2000 --nr_epochs 1000 --lrate 0.01 --output-model data_models/flow_model_moonsA.pt --output-train-data data_models/train_data_moonsA.pkl
```

Example 1: some mixture of three one-dimensional distributions
Example 2: Moons (from sklearn)

Above code will
* Sample `--n_train` data and save them in `--output-train_data` (`.pkl`)
* Train a flow model with params `--nr_epochs, --lrate, --dims`
* Save a trained model in `--output-model` (`.pt`)

# Sampling from flow 
(and plotting training data and sampled data)
```bash

python  flow_cnf_sample.py --n 2000 --input-model data_models/flow_model_1dA.pt --input-train-data data_models/train_data_1dA_n_3000.pkl 
python  flow_cnf_sample.py --n 2000 --input-model data_models/flow_model_moonsA.pt --input-train-data data_models/train_data_moonsA_n_2000.pkl 

```
Above code will
* Load training data from pickle `--input-train-data` and plot it
* Load trained flow model from `--input-mode`
* Sample `--n` samples from this flow model and plot it.

 
