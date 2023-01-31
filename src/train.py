# imports
import logging
import os
import time
import json
import numpy as np
import pickle

import torch

import models
import config
from utils import load_data, format_metrics


# -------------------- general setup --------------------
# args
args = config.get_args()

# reproducibility
np.random.seed(args.seed)
torch.manual_seed(args.seed)

if int(args.double_precision):
    torch.set_default_dtype(torch.float64)
if torch.cuda.is_available():
    torch.cuda.manual_seed(args.seed)
args.patience = args.epochs if not args.patience else int(args.patience)

# set device
device = 'cuda' if torch.cuda.is_available() else 'cpu'
args.device = device

# ------------------ setting up paths --------------------
config.LOGPATH = config.LOGPATH + '_' + args.data_name
if not os.path.exists(config.LOGPATH):
    os.makedirs(config.LOGPATH)

# ------------------ logging setup -----------------------
# https://www.electricmonk.nl/log/2017/08/06/understanding-pythons-logging-module/
logging.basicConfig(level=logging.DEBUG)
logFormatter = logging.Formatter("%(asctime)s || path: %(pathname)s || name: %(name)s || level: %(levelname)s|| \n %(message)s \n")
Logger = logging.getLogger(__name__)
Logger.setLevel(logging.DEBUG)

fileHandler = logging.FileHandler("{0}/{1}.txt".format(config.LOGPATH, config.LOGNAME))
fileHandler.setFormatter(logFormatter)
fileHandler.setLevel(logging.DEBUG)

consoleHandler = logging.StreamHandler()
consoleHandler.setFormatter(logFormatter)
consoleHandler.setLevel(logging.INFO)

Logger.addHandler(fileHandler)
Logger.addHandler(consoleHandler)

Logger.info('Using parameters: {}.'.format(args))

'''
Or:
logging.getLogger().setLevel(logging.INFO)
logging.basicConfig(level=logging.INFO,
                    handlers=[
                        logging.FileHandler(os.path.join(save_dir, 'log.txt')),
                        logging.StreamHandler()
                    ])
and working with logging (like logging.info(blabla))
'''



# -------------------------- init model -----------------------
model = None
# if args.lmf: 
#     raise NotImplementedError
#     # with open(args.lmf,'rb') as f:
#     #     model = pickle.load(f) #------------------------> do i need more to set up the model?
# else:
#     model = models.SpGAT(args)

# logging.info(str(model))
# tot_params = sum([np.prod(p.size()) for p in model.parameters()])
# logging.info(f"Total number of parameters: {tot_params}")

# model = model.to(device)

# optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, 
#                              weight_decay=args.weight_decay)
# if not args.lr_reduce_freq:
#         args.lr_reduce_freq = args.epochs
# lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer,
#                                                step_size=int(args.lr_reduce_freq),
#                                                gamma=float(args.gamma))


# ------------------------ iterate over different snapshots -------------------------
for root, _, files in os.walk(args.data):
    if 'climate_full' in args.data_name:
        files = sorted(files)  # in ascending order -- ok fro climate data, does not sort well for numbers as it considers strings
    elif 'synthetic800' in args.data_name:
        files = sorted(files, key=lambda x: int(x.split('_')[1].split('.pkl')[0]))  # for synth800 data
    else:
        raise ValueError('args.data_name not known.')
    args.current_time = 1  #if files are not consecutive, better to extract this from file name which is based on date
    args.flag = 0  # computer static loss only until we build enough history for args.lookback to work 
    for file in files:
        name = file.split('.pkl')[0]
        save_path = os.path.join(config.LOGPATH,name)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        ############## load data ####################
        # data_train_batches, data_val_batches, data_test_batches = load_data(file)
        data = load_data(os.path.join(root,file), args)
        args.nfeat = data['features'].shape[1]
        for x in data:
            if torch.is_tensor(data[x]):
                data[x] = data[x].to(device)
        # initialize model in the first step
        if not model:
            # if args.lmf: 
            #     raise NotImplementedError
            #     # with open(args.lmf,'rb') as f:
            #     #     model = pickle.load(f) #------------------------> do i need more to set up the model?
            # else:
            model = models.SpGAT(args)

            logging.info(str(model))
            tot_params = sum([np.prod(p.size()) for p in model.parameters()])
            logging.info(f"Total number of parameters: {tot_params}")

            model = model.to(device)

            optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, 
                                        weight_decay=args.weight_decay)
            # if not args.lr_reduce_freq:
            #         args.lr_reduce_freq = args.epochs
            # lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer,
            #                                             step_size=int(args.lr_reduce_freq),gamma=float(args.gamma))
        ############## iterate over epochs (not implemented for batches) ###############
        logging.info(f"################## {file} ###############")
        elapsed = time.time()
        counter = 0
        best_val_metrics = model.init_metrics()
        best_test_metrics = None
        best_emb = None
        for epoch in range(args.epochs):
            t = time.time()
            # set model in training mode
            model.train()
            optimizer.zero_grad()
            # forward pass
            embeddings = model.encode(data['features'], data['adj_train'])
            # get metrics including loss
            if args.current_time > args.lookback:
                args.flag = 1
            train_metrics = model.compute_metrics(data, embeddings, args, split='train')
            # compute gradient over the loss (backprop) 
            train_metrics['loss'].backward()
            optimizer.step()
            # lr_scheduler.step()


            # ----- optimization update in epoch is finished, taking care of prints
            # if (epoch + 1) % args.log_freq == 0:
            #     logging.info(" ".join(['Epoch: {:04d}'.format(epoch + 1),
            #                         'lr: {}'.format(lr_scheduler.get_last_lr()[0]),
            #                         format_metrics(train_metrics, 'train'),
            #                         'time: {:.4f}s'.format(time.time() - t)
            #                         ]))
            if (epoch + 1) % args.log_freq == 0:
                logging.info(" ".join(['Epoch: {:04d}'.format(epoch + 1),
                                    'lr: {}'.format(args.lr),
                                    format_metrics(train_metrics, 'train'),
                                    'time: {:.4f}s'.format(time.time() - t)
                                    ]))
            if (epoch + 1) % args.eval_freq == 0:
                model.eval()
                embeddings = model.encode(data['features'], data['adj_train'])
                val_metrics = model.compute_metrics(data, embeddings, args, split='val')
                if (epoch + 1) % args.log_freq == 0:
                    logging.info("   "+" ".join(['Epoch: {:04d}'.format(epoch + 1), format_metrics(val_metrics, 'val')]))
                if model.has_improved(best_val_metrics, val_metrics):
                    best_test_metrics = model.compute_metrics(data, embeddings, args, split='test')
                    best_emb = embeddings.cpu()
                    np.save(os.path.join(save_path, '{0}_{1}.npy'.format(config.MODEL_NAME ,args.current_time)), best_emb.detach().numpy())
                    best_val_metrics = val_metrics
                    counter = 0
                else:
                    counter += 1
                    if counter == args.patience and epoch > args.min_epochs:
                        logging.info("Early stopping")
                        break

        # ----------- optimization finished; taking care of prints
        logging.info("Optimization Finished!")
        logging.info("Total time elapsed: {:.4f}s".format(time.time() - elapsed))
        if args.tmod == 'S':
            if not best_test_metrics:
                model.eval()
                best_emb = model.encode(data['features'], data['adj_train'])
                best_test_metrics = model.compute_metrics(data, best_emb, args, split='test')
            logging.info(" ".join(["Val set results:", format_metrics(best_val_metrics, 'val')]))
            logging.info(" ".join(["Test set results:", format_metrics(best_test_metrics, 'test')]))

        # --- update history by nodes present in this time (data[node_list])
        best_emb = best_emb.cpu().detach().numpy()
        for node in data['node_list'].tolist():
            if node in config.HISTORY:
                if len(config.HISTORY[node]) >= args.lookback:
                    config.HISTORY[node].pop(-1)
                config.HISTORY[node] = [(args.current_time, best_emb[node,:])] + config.HISTORY[node]
            else:
                config.HISTORY[node] = [(args.current_time, best_emb[node,:])]
        
        # ---save
        np.save(os.path.join(save_path, '{0}_{1}.npy'.format(config.MODEL_NAME, args.current_time)), best_emb)
        json.dump(vars(args), open(os.path.join(save_path, '{0}_{1}.json'.format(config.CONFIG_NAME, args.current_time)), 'w'))
        torch.save(model.state_dict(), os.path.join(save_path, '{0}_{1}.path'.format(config.MODEL_NAME, args.current_time)))
        with open(os.path.join(save_path, 'nodes_{0}.pkl'.format(args.current_time)), 'wb') as f:
            pickle.dump(data['node_list'].cpu().detach().numpy(), f)
        logging.info(f"Saved model in {save_path}")

        # ---- update current time for the next file
        args.current_time += 1
