
import torch
import torch.nn as nn
import torch.nn.functional as F
from layers import SpGraphAttentionLayer
from sklearn.metrics import accuracy_score, f1_score
import config
import numpy as np


class BaseModel(nn.Module):
    def __init__(self, args):
        super(BaseModel, self).__init__()

    def encode(self, x, adj):
        raise NotImplementedError

    def compute_metrics(self, emb, data):
        '''
        This function is used for training, evaluation, and testing
        '''
        raise NotImplementedError
        
    def init_metrics(self):
        raise NotImplementedError
    
    def has_improved(self, m1, m2):
        raise NotImplementedError



class BLE(BaseModel):
    def __init__(self, args):
        super(BLE, self).__init__()

    def encode(self, x, adj):
        raise NotImplementedError

    def compute_metrics(self, emb, data):
        '''
        This function is used for training, evaluation, and testing
        '''
        raise NotImplementedError
        metrics = {'loss': loss, 'acc': acc, 'f1': f1}
        return metrics
    
    def init_metrics(self):
        raise NotImplementedError
        return {'acc': -1, 'f1': -1}
    
    def has_improved(self, m1, m2):
        raise NotImplementedError
        return m1["f1"] < m2["f1"]




class SpGAT(nn.Module):
    def __init__(self, args): #nfeat, nhid, emb_dim, dropout, alpha, nheads):
        """Sparse version of GAT. From pyGAT repo."""
        super(SpGAT, self).__init__()
        self.dropout = args.dropout
        self.mode = args.tmod
        self.attentions = [SpGraphAttentionLayer(args.nfeat, 
                                                 args.nhid, 
                                                 dropout=args.dropout, 
                                                 alpha=args.alpha, 
                                                 concat=True) for _ in range(args.nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        # for supervised task (classification), args.dim = number of classes
        self.out_att = SpGraphAttentionLayer(args.nhid * args.nheads, 
                                                args.dim, 
                                                dropout=args.dropout, 
                                                alpha=args.alpha, 
                                                concat=False)
        if self.mode == 'S':
            if args.dim > 2:
                self.f1_average = 'micro'
            else:
                self.f1_average = 'binary'

    def encode(self, x, adj):
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x, adj) for att in self.attentions], dim=1)
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.elu(self.out_att(x, adj))
        return x #F.log_softmax(x, dim=1)

    def compute_metrics(self, data, emb, args, split):
        '''
        This function is used for training, evaluation, and testing
        '''
        
        if self.mode == 'U':
            loss = unsupervised_loss(emb=emb, data=data, args=args)
            metrics = {'loss': loss}  #, 'acc': acc, 'f1': f1}
        else:  # self.mode == 'S'
            idx = data[f'idx_{split}']
            output = F.log_softmax(emb[idx], dim=1)
            loss = F.nll_loss(output, data['labels'][idx])
            acc, f1 = acc_f1(output, data['labels'][idx], average=self.f1_average)
            metrics = {'loss': loss, 'acc': acc, 'f1': f1}
        return metrics
    
    def init_metrics(self):
        if self.mode == 'U':
            return {'loss': 100}
        return {'acc': -1, 'f1': -1}
    
    def has_improved(self, m1, m2):
        if self.mode == 'U':
            return m1["loss"] > m2["loss"]
        return round(m1['f1'],1) < round(m2['f1'],1)




def unsupervised_loss(emb, data, args):
    '''
    if flag = 1: compute both static and dynamic loss and return average of those
    else: computer only static loss (i.e., not enough historical info)
    '''
    # make emb' as exclusive embedding that excludes nodes that are not present in this timestamp (no edge)
    # node_list is in the same order as emb' rows will be
    emb_current = emb.index_select(dim=0, index=data['node_list'])
    # static loss
    sloss = static_loss(data, emb_current, args)
    if args.flag:
        # dynamic loss
        dloss = dynamic_loss(data, emb_current, args)
        if dloss:
            return (sloss + dloss)/2
    return sloss

def dynamic_loss(data, emb, args):
    '''
    adj -> torch tensor
    emb -> torch tensor
    current_time -> int
    weight decay = W --> W**0.5 = w
    return: loss = Normalized(sigma_overNodes sigma_overLookback ||wX - wX'||^2_2) 
            and X=embedding_current X'=embedding_old
    Assumptions: 
        - adj matrix is normalized to have values between 0 and 1
        - the weight decay is the same for each row
    So the normalization factor for loss is 1/sqrt(N)sqrt(sum(weights_decay))
    '''
    # iterate over lookback-length history of each node in node_list
    loss = 0
    count = 0
    for lb in range(args.lookback):
        # make diagonal matrix of exponential decay coefficients (coef = 0 if node history not available)
        # [(coef_outside_norm,coef_inside_norm,embedding_vector)]
        coefs_emb = [(weight_decay(args.current_time,config.HISTORY[node][lb][0],1),
                    weight_decay(args.current_time,config.HISTORY[node][lb][0],2),
                    config.HISTORY[node][lb][1]) if len(config.HISTORY[node])>lb \
                        else (0,0,np.asarray([])) for node in data['node_list'].tolist()]
        # coefs inside norm (the exponent was divided by 2)
        coefs =  torch.diag(torch.FloatTensor([i[1] for i in coefs_emb]))
        # sum of coefs outside norm (exponent not divided by 2)
        coefs_sum = sum([i[0] for i in coefs_emb])
        # if none of the nodes has been seen before, don't computer dynamic loss and only consider static loss
        if coefs_sum == 0:  
            return None
        emb_old = [i[2] for i in coefs_emb] # this is the old embedding 
        # padding the embedding for nodes that don't have the history
        max_cols = max([len(i) for i in emb_old])
        emb_old = [np.append(i , np.asarray([[0] * (max_cols)])) if len(i)==0 else i for i in emb_old]
        emb_old = torch.FloatTensor(np.asarray(emb_old)).to(args.device)
        # updating the emb_exc and vecs by adding weight to them
        coefs = coefs.to(args.device)
        emb = torch.mm(coefs,emb)
        emb_old = torch.mm(coefs,emb_old) 
        loss += (1/((emb.shape[0]**0.5)*(coefs_sum**0.5)))*torch.linalg.matrix_norm(emb - emb_old, 'fro')
        count +=1 
    return loss/count

def weight_decay( tc, ti, w):
    '''
    get exponential weight decay for tc (current time) and ti (time in the past) weighted by w
    '''
    return np.exp((ti-tc)/w) # this will be used inside the |emb1-emb2|**2, so divided by two is for canceling the poewr 2
    # return np.exp((ti-tc)/2w) # for considering further back history too

def static_loss(data, emb, args):
    '''
    emb and adj --> torch tensor
    return: loss = dissimilarity between ones and similarity between zeros of adj in emb space
    '''
    adj = data['adj_train']
    adj = adj.index_select(dim=0, index=data['node_list']).index_select(dim=1, index=data['node_list'])
    # if we directly worked with adj and emb, the many nodes that have not happened in this timestamp would
    #       bias the zero_sample. If we don't use zero sample, then we can directly work with emb and adj.

    # sample all 1 in adj and same amount of 0s
    sample_size = args.sample_size
    # sample ones
    ones_idx = adj.nonzero()
    if sample_size < ones_idx.shape[0]:
        ones_sample_idx = np.random.choice(ones_idx.shape[0],sample_size,replace=False)
        ones_idx = ones_idx[ones_sample_idx]
    # sample zeros
    zeros_idx = (adj == 0).nonzero()
    if sample_size < zeros_idx.shape[0]: 
        zeros_sample_idx = np.random.choice(zeros_idx.shape[0],sample_size,replace=False)
        zeros_idx = zeros_idx[zeros_sample_idx]
    # final loss
    ones_loss = torch.nn.functional.cosine_similarity(emb[ones_idx[:,0],:],  emb[ones_idx[:,1],:]).mean()
    zeros_loss = torch.nn.functional.cosine_similarity(emb[zeros_idx[:,0],:],  emb[zeros_idx[:,1],:]).abs().mean()
    loss = (zeros_loss - ones_loss) + 1  #  worst case: 1 - 0 + 1 = 2, best case: 0 - 1 + 1 =0
    return loss

    
def acc_f1(output, labels, average='binary'):
    preds = output.max(1)[1].type_as(labels)
    if preds.is_cuda:
        preds = preds.cpu()
        labels = labels.cpu()
    accuracy = accuracy_score(preds, labels)
    f1 = f1_score(preds, labels, average=average)
    return accuracy, f1