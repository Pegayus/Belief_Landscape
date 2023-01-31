# Dynamic Belief Landscape Modeling using Attention Graph Neural Network  
This is a dynamic attention-based model for learning the belief landscape from a set of tweets in a period of time. 

At each timestamp, the current version uses user retweet relationship (user-user) as the main adjacency matrix and the user hashtag relationship (user-hashtag) as the feature vector for each user (note that it is possible to not use the features using flags available in `config.py`). You can define the user-user adjacency based on any relationship of preference, for example based on the similarity of the hashtags they use. But, you have to maintain it as an adjacency matrix (i.e., #columns=#rows). The only matrix that can be non-square in the input is the feature matrix.

The bird's-eye-view of the repository is as follows,
```
* src
    * train.py: The main script to run for training the model. 
    * config: The input parameters for running the train script.
    * models.py: The models used for training. The current model is a modified version of sparse attention graph neural networks that accomodoated dynamic training with proper loss functions.
    * layers.py: The neural network layers used by the model.
    * utils.py: All the function for loading and preprocessing data before feeding to network are available here.
* data
    * climate_sample: A sample of the climate data used.
* logs: The result of the training for each dataset and timestamp is saved under this directory. You can change this defult setting in config file.
```


# Training
Run the below command with desired configuration using flags in `config.py`. An example:
```
python train.py --data <path-to-data> --data_name <name-to-save-with> --feat 1 --tmod S --epochs 1000 --patience 50 --dropout 0.1 
```
The training at each timestamp starts with loading the respective adjacency and feature matrices. The results for that timestamp are then saved under ./logs directory before the training moves onto the next timestamp. At the end of each timestamp training, the best embedding is saved under the `log` directory.

The model can be trained both in supervised and unsupervised mode. In supervised mode, the number of classes and labels should be available in the input data (see sample data explained in [Input Data](#input-data)).


# Input Data
A sample of the data used in training can be found under `./data/climate_sample`. I have used this code to train synthetic data (8000 agents) and climate data (full and filtered versions of it). For all data, I used the structure explained below.

The data is split into daily timestamps in a folder which will be given as input with `--data` argument (e.g., the folder named `adj_feat_wlabel` in the sample data under `./data/climate_sample`). Under this folder and for each timestamp, adjacency and feature matrices are saved as a dictionary into a pickled file. 

As all nodes (users) do not repeat in every timestamp, it is important to build the adjacency and feature matrices using the same order for all timestamps. In this way, whenever the data is loaded for a timestamp, the data loader can easily determine for the trainer which rows of the adjacency and feature matrix to use (e.g., the order for both users and features that determines the order of adjacency matrix rows and feature matrix columns is accessible under `./data/climate_sample/adj_node_order_and_feat_order.pkl`).

