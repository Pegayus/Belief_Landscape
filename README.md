# Dynamic Belief Landscape Modeling with Attention Graph Neural Network  

## Introduction
This repository introduces a dynamic attention-based model designed to learn the belief landscape from a dataset comprising a series of tweets over a defined period. The dataset utilized for this project is credited to Dr. Joshua E. Introne from Syracuse University, and the study draws its inspiration from his paper, [**Measuring Belief Dynamics on Twitter**](https://ojs.aaai.org/index.php/ICWSM/article/view/22154). 

In its current iteration, the model leverages the user retweet relationship to form the primary adjacency matrix, and the user-hashtag relationship to create the feature vector for each user. Note, however, that the model's configuration can be altered to exclude features using the flags available in `config.py`.

While the user-user adjacency can be defined based on any preferred relationship, such as the similarity of hashtags used, the adjacency matrix needs to be maintained (i.e., #columns = #rows). The only non-square matrix permissible in the input is the feature matrix.

## Repository Structure
The repository is structured as follows:

```
* src
    * train.py: The primary script for training the model. 
    * config: Contains the input parameters necessary for running the train script.
    * models.py: Houses the models utilized for training. The current model is a modified version of sparse attention graph neural networks, accommodating dynamic training with appropriate loss functions.
    * layers.py: Contains the neural network layers utilized by the model.
    * utils.py: Consists of all the functions for loading and preprocessing data before network feeding.
* data
    * climate_sample: A sample of the climate data used for training.
* logs: The training results for each dataset and timestamp are saved in this directory. This default setting can be changed in the config file.
```

## Training the Model
To train the model, use the following command, substituting in your preferred configuration flags from `config.py`:

```
python train.py --data <path-to-data> --data_name <name-to-save-with> --feat 1 --tmod S --epochs 1000 --patience 50 --dropout 0.1 
```

At the start of each timestamp, the relevant adjacency and feature matrices are loaded, with the results saved in the `./logs` directory before training advances to the next timestamp. The best embedding is stored in the `log` directory at the end of each timestamp's training.

The model supports both supervised and unsupervised training modes. In supervised mode, the number of classes and labels must be included in the input data (refer to the sample data explained in the [Input Data](#input-data) section).

## Input Data
A sample of the data used for training can be found under `./data/climate_sample`. The structure of this data has been used to train synthetic data (8000 agents) and climate data (both full and filtered versions).

The data is split into daily timestamps in a folder provided as input using the `--data` argument. For each timestamp, the adjacency and feature matrices are stored as a dictionary in a pickled file.

As not all nodes (users) reappear in every timestamp, it's crucial to construct the adjacency and feature matrices using the same order across all timestamps. This allows the data loader to identify which rows of the adjacency and feature matrix to use when the data is loaded for a timestamp. The order for both users and features, which determines the order of adjacency matrix rows and feature matrix columns, can be found under `./data/climate_sample/adj_node_order_and_feat_order.pkl`.
