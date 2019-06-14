#!/usr/bin/env python

import numpy as np
import time
import argparse
import os
import torch
import torch.nn as nn
from torch import optim
from LSTM import lstm
from pytorch_data_loader import Dataset, util_collate_fn
from util import plotter, show_prog, save_prog, evaluate_sepsis_score
from torch.utils.data import DataLoader

#TODO: make this part a bit nicer
train_path = '/home/osvald/Projects/Diagnostics/CinC_data/w_tensors/'
save_path = '/home/osvald/Projects/Diagnostics/Sepsis/Models/lstm/'


######## __GENERAL__ ########
parser = argparse.ArgumentParser(description='training control')
parser.add_argument('--disable-cuda', action='store_true',
                    help='Disable CUDA')
parser.add_argument('-epochs', action='store', default=50, type=int,
                    help='num epochs')
parser.add_argument('-batch', action='store', default=256, type=int,
                    help='batch size')
parser.add_argument('-nosave', action='store_true',
                    help='do not save flag')
parser.add_argument('-prog', action='store_true',
                    help='show progress')

######## __LSTM__ ########
parser.add_argument('-embedding', action='store', default=64, type=int,
                    help='embedding size')
parser.add_argument('-hidden', action='store', default=64, type=int,
                    help='lstm hidden dim')
parser.add_argument('-layers', action='store', default=2, type=int,
                    help='num lstm layers')
parser.add_argument('-fcl', action='store', default=0, type=int,
                    help='fully connected size')
parser.add_argument('-drop', action='store', default=0.25, type=float,
                    help='droprate')

######## __OPTIM__ ########
parser.add_argument('-lr', action='store', default=1, type=float,
                    help='learning rate')
parser.add_argument('-momentum', action='store', default=0, type=float,
                    help='momentum')
parser.add_argument('-gamma', action='store', default=0.1, type=float,
                    help='learning rate decay')
parser.add_argument('-patience', action='store', default=5, type=int,
                    help='patience for lr decay')
args = parser.parse_args()

model_name = '_'.join(['b'+str(args.batch), str(args.embedding), str(args.layers)+'x'+str(args.hidden),
                       str(args.fcl), 'd'+str(args.drop), 'lr'+str(args.lr), 'm'+str(args.momentum),
                       'g'+str(args.gamma), 'p'+str(args.patience)])

# Create target Directory if don't exist
if not os.path.exists(save_path+model_name):
    os.mkdir(save_path+model_name)
elif not args.nosave:
    print('WARNING: overwriting existing directory:', model_name)
save_path = save_path + model_name + '/'
if args.nosave: print('WARNING: MODEL AND DATA ARE NOT BEING SAVED')

######## __GPU_SETUP__ ########
if not args.disable_cuda and torch.cuda.is_available():
    args.device = torch.device('cuda')
    torch.set_default_tensor_type('torch.cuda.DoubleTensor')
else:
    args.device = torch.device('cpu')
    torch.set_default_tensor_type('torch.DoubleTensor')



epochs = args.epochs
batch_size = args.batch
save_rate = 5
save = not args.nosave

load_model = False #'/home/osvald/Projects/Diagnostics/Sepsis/Models/lstm/32_2x64_l01_p25/model_epoch80'
load_folder = False #'/home/osvald/Projects/Diagnostics/Sepsis/Models/lstm/32_2x64_l01_p25'
offset = 0

n = 40336
n=20000
split = 0.9
ind = list(range(n))
div = int(n * split)
partition = dict([])
partition['train'] = list(ind[:div])
partition['validation'] = list(ind[div:n])



model = lstm(embedding=args.embedding, hidden_size=args.hidden, fcl=args.fcl, num_layers=args.layers, 
             batch_size=batch_size, fcl_out=bool(args.fcl), embed=bool(args.embedding), droprate=args.drop).to(args.device)
criterion = nn.BCEWithLogitsLoss(pos_weight=torch.DoubleTensor([1.8224]).to(args.device), reduction='none')
optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, verbose=False, factor=args.gamma, patience=args.patience)

if load_model:
    model.load_state_dict(torch.load(load_model))
    train_losses = np.concatenate((np.load(load_folder +'/train_losses.npy'), np.zeros(epochs)))
    train_pos_acc = np.concatenate((np.load(load_folder +'/train_pos_acc.npy'), np.zeros(epochs)))
    train_neg_acc = np.concatenate((np.load(load_folder +'/train_neg_acc.npy'), np.zeros(epochs)))
    val_losses = np.concatenate((np.load(load_folder +'/val_losses.npy'), np.zeros(epochs)))
    val_pos_acc = np.concatenate((np.load(load_folder +'/val_pos_acc.npy'), np.zeros(epochs)))
    val_neg_acc = np.concatenate((np.load(load_folder +'/val_neg_acc.npy'), np.zeros(epochs)))
    utility = np.concatenate((np.load(load_folder +'/utility.npy'), np.zeros(epochs)))
    plotter(model_name, utility, train_losses, train_pos_acc, train_neg_acc,
        val_losses, val_pos_acc, val_neg_acc)
else:
    train_losses = np.zeros(epochs)
    val_losses = np.zeros(epochs)
    # train accuracy
    train_pos_acc = np.zeros(epochs)
    train_neg_acc = np.zeros(epochs)
    # val accuracy
    val_pos_acc = np.zeros(epochs)
    val_neg_acc = np.zeros(epochs)
    # utility
    utility = np.zeros(epochs)


train_data = Dataset(partition['train'], train_path, util_weights=True)
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, collate_fn=util_collate_fn)

val_data = Dataset(partition['validation'], train_path, util_weights=True)
val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=True, collate_fn=util_collate_fn)


def lstm_util_train():
    running_loss, pos_total, pos_correct, neg_total, neg_correct  = 0, 0, 0, 0, 0
    for batch, labels, util, seq_len in train_loader:
        # pass to GPU if available
        batch, labels, util = batch.to(args.device), labels.to(args.device), util.to(args.device)
        max_len = labels.shape[1]

        optimizer.zero_grad()
        outputs = model(batch, seq_len, max_len)
        outputs = outputs.view(-1, max_len)
        loss = torch.mean(criterion(outputs, labels) * util)
        loss.backward()
        optimizer.step()
        running_loss += loss.cpu().data.numpy()
    
        # Train Accuracy
        for i in range(labels.shape[0]):
            targets = labels.data[i,:int(seq_len[i])].cpu().numpy()
            prediction = torch.round(torch.sigmoid(outputs.data[i, :int(seq_len[i])])).cpu().numpy()
            match = targets == prediction
            pos_total += (targets == 1).sum()
            neg_total += (targets == 0).sum()
            pos_correct += (match * (targets == 1)).sum()
            neg_correct += (match * (targets == 0)).sum()
    
    train_losses[epoch] = running_loss/len(train_loader) * 100 # scaling factor
    train_pos_acc[epoch] = pos_correct/pos_total
    train_neg_acc[epoch] = neg_correct/neg_total

def lstm_util_test():
    predictions, truth = [], []
    running_loss, pos_total, pos_correct, neg_total, neg_correct  = 0, 0, 0, 0, 0
    with torch.set_grad_enabled(False):
        for batch, labels, util, seq_len in val_loader:
            # pass to GPU if available
            batch, labels = batch.to(args.device), labels.to(args.device)
            max_len = labels.shape[1]

            outputs = model(batch, seq_len, max_len, batch_size)
            outputs = outputs.view(-1, max_len)
            loss = torch.mean(criterion(outputs, labels) * util)
            running_loss += loss.cpu().data.numpy()
            
            # Validation Accuracy
            for i in range(labels.shape[0]):
                targets = labels.data[i,:int(seq_len[i])].cpu().numpy()
                prediction = torch.round(torch.sigmoid(outputs.data[i, :int(seq_len[i])])).cpu().numpy()
                match = targets == prediction
                pos_total += (targets == 1).sum()
                neg_total += (targets == 0).sum()
                pos_correct += (match * (targets == 1)).sum()
                neg_correct += (match * (targets == 0)).sum()
                # remove label extension
                predictions.append(prediction)
                if targets.sum() > 0:
                    onset = np.argwhere(targets == 1)[0][0] 
                    targets[onset:onset+6] = 0 

                truth.append(targets)

        val_losses[epoch] = running_loss/len(val_loader) * 100 # scaling applied at end for ease of reading graph
        val_pos_acc[epoch] = pos_correct/pos_total
        val_neg_acc[epoch] = neg_correct/neg_total
        utility[epoch] = evaluate_sepsis_score(truth, predictions)

'''__main__'''
start = time.time()
for epoch in range(offset, offset + epochs):
    model.train()
    lstm_util_train()
    model.eval()
    lstm_util_test()
    scheduler.step(train_losses[epoch]) 

    if args.prog:
        show_prog(epoch, utility[epoch], train_losses[epoch], val_losses[epoch], train_pos_acc[epoch],
              train_neg_acc[epoch], val_pos_acc[epoch], val_neg_acc[epoch], (time.time() - start))
    
    if save:
        save_prog(model, save_path, utility, train_losses, val_losses, train_pos_acc,
              train_neg_acc, val_pos_acc, val_neg_acc, epoch, save_rate)
    

# PLOT GRAPHS
if save:
    plotter(model_name, utility, train_losses, train_pos_acc, train_neg_acc,
            val_losses, val_pos_acc, val_neg_acc, save=save_path, show=False)
else:
    plotter(model_name, utility, train_losses, train_pos_acc, train_neg_acc,
            val_losses, val_pos_acc, val_neg_acc, save=False, show=True)

print('Model:', model_name, 'completed ; ', epochs, 'epochs', 'in %ds' % (time.time()-start))
print('max utility: %0.3f at epoch %d' % (max(utility), utility.argmax()+1))