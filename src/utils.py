import torch
import pickle
import scipy.sparse as sp
import numpy as np
import config
import pdb


def load_data(file, args):
    '''
    Loads data similar to the link prediction task.
    Assumption:
        - In supervised task, the classes are labeld as 0,1,2,...
    '''
    with open(file, 'rb') as f:
        data = pickle.load(f)

    adj = data['adj']  # csr matrix
    nodes = range(adj.shape[0])
    if len(config.HISTORY) == 0:  # initialize the history map for the first reading of data
        config.HISTORY = {n:[] for n in nodes}
    node_list = process(adj)  # nodes present at this time

    # normalize adj matrix so values are between 0 and 1
    adj = normalize_adj(adj + sp.eye(adj.shape[0]))
    if args.feat:
        features = data['feat']
        features = normalize_features(features)
    else:
        features = sp.eye(adj.shape[0])  # --> will eventually be dense and to troch float 

    if args.tmod == 'S':
        args.dim = data['num_classes']
        node_label = data['label']  # {node index: class label}
        node_label = {k:v for k,v in node_label.items() if k in node_list}
        all_labeled_idx = list(node_label)
        labels = np.zeros((adj.shape[0],))
        labels[all_labeled_idx] = np.asarray([node_label[i] for i in all_labeled_idx])
        idx_train, idx_val, idx_test = split_data(labels_idx=all_labeled_idx,
                                                    seed=args.split_seed, 
                                                    val_prop=0.15, test_prop=0.15)
        labels = torch.LongTensor(labels)

    # for LP task : train, val, test (take care of existing nodes)
    # adj_train, train_edges, train_edges_false, val_edges,\
    #     val_edges_false, test_edges, test_edges_false = mask_edges(adj, args)

    # make torch tensors
    adj = torch.FloatTensor(adj.toarray())  # dense matrix
    features = torch.FloatTensor(features.todense())  # dense matrix   < -------- the process is killed here due to memory stuff
    node_list = torch.LongTensor(node_list)

    # return {'adj_train': adj, 'features': features, 'train_edges': train_edges,
        #  'train_edges_false': train_edges_false,
        #  'val_edges': val_edges, 'val_edges_false': val_edges_false, 
        #  'test_edges': test_edges, 'test_edges_false': test_edges_false}
    if args.tmod == 'S':
        return dict(adj_train = adj, features = features, node_list = node_list, labels=labels,
                idx_train = idx_train, idx_val = idx_val, idx_test = idx_test)
    return dict(adj_train = adj, features = features, node_list = node_list)
    

def split_data(labels_idx, val_prop, test_prop, seed):
    np.random.seed(seed)
    np.random.shuffle(labels_idx)
    nb_val = round(val_prop * len(labels_idx))
    nb_test = round(test_prop * len(labels_idx))
    idx_val, idx_test, idx_train = labels_idx[:nb_val], labels_idx[nb_val:nb_val + nb_test], labels_idx[nb_val + nb_test:]
    return idx_train, idx_val, idx_test

def process(adj):
    # existing edges --> existing nodes
    # x, y = sp.triu(adj).nonzero()  # for undirected adj 
    x, y = adj.nonzero()  # indices (row,col) of non zero elements 
    existing_nodes = list(set(x).union(set(y)))
    return np.asarray(existing_nodes)
  
def normalize_adj(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv_sqrt = np.power(rowsum, -0.5).flatten()
    r_inv_sqrt[np.isinf(r_inv_sqrt)] = 0.
    r_mat_inv_sqrt = sp.diags(r_inv_sqrt)
    return mx.dot(r_mat_inv_sqrt).transpose().dot(r_mat_inv_sqrt)

def normalize_features(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1.).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx  

def mask_edges(adj, args):
    '''
    input: adj as csr matrix, input arguments
    output: 
        - adj_train: adj used for training, excluding edges in val and test set
        - edges_train, edges_val, edges_test: set of existing edges for train, val, and test sets
        - non_edges_train, non_edges_val, non_edges_test: set of non-edges for train, val, and test sets 
    * considers the eixsting nodes (those with at least one non-zero element in its row)
    '''
    np.random.seed(args.seed)
    # get positive edges 
    # x, y = sp.triu(adj).nonzero()  # for undirected adj 
    x, y = adj.nonzero()  # indices (row,col) of non zero elements 
    pos_edges = np.array(list(zip(x, y)))  
    np.random.shuffle(pos_edges)
    # get number of test and value edges needed for each
    n_pos = len(pos_edges)
    n_val = int(n_pos * args.val_prop)
    n_test = int(n_pos * args.test_prop)
    # get negative edges
    existing_nodes = list(set(x).union(set(y)))
    mapper = dict(enumerate(existing_nodes))
    adj_filter = adj.toarray().take(existing_nodes, axis=0).take(existing_nodes, axis=1)
    neg_adj = sp.csr_matrix(1. - adj_filter)
    x , y = neg_adj.nonzero()
    neg_edges = sampler(x,y,mapper,size=n_val+n_test+1)  # np.array and already shuffled
    # neg_edges = np.asarray(list(zip(x,y)))  # if nodes existed in all timestamps
    # neg_edges = np.array([i for i in zip(x, y) if i[0] in existing_nodes and i[1] in existing_nodes])
    # np.random.shuffle(neg_edges)
    # get the val, tes, train edges respectively
    val_edges, test_edges, train_edges = pos_edges[:n_val], pos_edges[n_val:n_test + n_val], pos_edges[n_test + n_val:]
    val_edges_false, test_edges_false = neg_edges[:n_val], neg_edges[n_val:n_test + n_val]
    train_edges_false = np.concatenate([neg_edges, val_edges, test_edges], axis=0)  # not used in training
    # build adj for training
    adj_train = sp.csr_matrix((np.ones(train_edges.shape[0]), (train_edges[:, 0], train_edges[:, 1])), shape=adj.shape)
    # adj_train = adj_train + adj_train.T  # if undireced and used x,y = sp.triu()
    return adj_train, torch.LongTensor(train_edges), torch.LongTensor(train_edges_false), torch.LongTensor(val_edges), \
           torch.LongTensor(val_edges_false), torch.LongTensor(test_edges), torch.LongTensor(
            test_edges_false)  

def sampler(x,y,mapper,size):
    '''
    Take (x,y) indices and pick randomly of size "size" and map x and y by mapper
    '''
    idx = range(len(x))
    sampled_idx = np.random.choice(idx, size)
    return np.asarray([(mapper[x[i]],mapper[y[i]]) for i in sampled_idx])


def format_metrics(metrics, split):
    """Format metric in metric dict for logging."""
    return " ".join(
            ["{}_{}: {:.4f}".format(split, metric_name, metric_val) for metric_name, metric_val in metrics.items()])
