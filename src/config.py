from datetime import datetime
import argparse


# ----------------------- global vars -----------------------------

# save output messages including args (data and hyperparameters)
LOGPATH = '../logs/' + datetime.now().strftime('%Y%m%d_%H%M%S')
LOGNAME = 'log'

# save model.pkl (weights), embedding.npy, config.json
MODEL_NAME = 'model'
EMBEDING_NAME = 'embedding'
CONFIG_NAME = 'config'

# history keeps the last args.lookback embedding of each node in a dict: {node_i:[(t,emb_vec),(t,emb_vec),...]}
HISTORY_NAME = 'history'
HISTORY = {}

# ------------------------------ parser ----------------------------


def get_args():
    parser = argparse.ArgumentParser()
    # data and general setup
    parser.add_argument('--lmf', default=None, type=str,
                        help='Load model from: if there is a pre-trained model, give the path here.')
    parser.add_argument('--data', default= '../data/climate_full/user_hashtag_combined_adj_unnormalized',
                        type=str, help='The directory with all timestamp files.')
    parser.add_argument('--data_name', default='climate_full', type=str, help='the data_name to create logpath for.')
    parser.add_argument('--feat', default=1 , type=int, help='use features for training (1) or not.')
    parser.add_argument('--seed', default=42, type = int, help='seed for initializign random-based operatos.')
    parser.add_argument('--split_seed', default=1234, type=int, help='seed for data splits (train/test/val) in supervised mode.')
    parser.add_argument('--double_precision', default=0, type=int, 
                        help='whether to use double precision (1) or not (0)')
    parser.add_argument('--val_prop', default=0.1, type=float,
                        help='proportion of edges and non-edges used for validation in link prediction task.')
    parser.add_argument('--test_prop', default = 0.1, type=float,
                        help='proportion of edges and non-edges used for test in link prediction task.')
    # training
    parser.add_argument('--tmod', default='U', type = str, choices=['U','S'],
                        help='The training mode. U: unsupervised, S: supervised or semisupervised')
    parser.add_argument('--epochs', default=5000, type=int,
                        help='Maximum number of epochs to train for.')
    parser.add_argument('--lr_reduce_freq', default=None, type=int,
                        help='reduce lr every lr_reduce_freq or None to keep lr constant')
    parser.add_argument('--gamma', default=0.5, type=float, help='gamma for lr scheduler')
    parser.add_argument('--patience', default=100, type=int, help='patience for early stopping')
    parser.add_argument('--log_freq', default=1, type=int, 
                        help='how often to print train/val metrics (in epochs)')
    parser.add_argument('--eval_freq', default=1, type=int, 
                        help='how often to compute val metrics (in epochs)')
    parser.add_argument('--min_epochs', default=100, type=int,  
                        help='do not early stop before min-epochs')

    # unsupervised loss
    # note that lookback does NOT mean we look at lookback steps across time window for all dataset
    # it means for each node, we take its lookback number of recent embeddings. 
    # so a node can last appear 10 timewindow ago and another node 2 timewindow ago, the time 10 and 2 for both is lookback=1
    parser.add_argument('--lookback', default= 3, type=int, 
                        help='the legnth of looking back in the recent hisotry of a node for dynamic loss.')
    parser.add_argument('--sample_size', default=1000, type=int, 
                        help='sample size for sampling 1s and 0s from adj in static loss.')
    # spGAT
    parser.add_argument('--nhid', type=int, default=8, help='Number of hidden units.')
    parser.add_argument('--nheads', type=int, default=8, help='Number of head attentions.')
    parser.add_argument('--dropout', type=float, default=0, help='Dropout rate (1 - keep probability).')
    parser.add_argument('--alpha', type=float, default=0.2, help='Alpha for the leaky_relu.')
    parser.add_argument('--dim', type=int, default=264, help='Final embedding dimension.')
    parser.add_argument('--weight_decay', type=float, default=0.3, help='weight decay for optimizer.')
    parser.add_argument('--lr', type=float, default=1e-03, help='learning rate for optimizer.')
    return parser.parse_args()





