#!/usr/bin/env python

import numpy as np
import time
import argparse
import torch
import torch.nn as nn
from torch import optim
from model import lstm, TCN
#from model_attention import LSTM_attn
from pytorch_data_loader import Dataset, collate_fn, util_collate_fn
from driver import save_challenge_predictions
from util import plotter, show_prog, save_prog, evaluate_sepsis_score
from torch.utils.data import DataLoader

#TODO: make this a bit nicer
#data_path = 'C:/Users/Osvald/Sepsis_ML/'
#data_path = '/home/osvald/Projects/Diagnostics/CinC_data/tensors/'
train_path = '/home/osvald/Projects/Diagnostics/CinC_data/w_tensors/'
save_path = '/home/osvald/Projects/Diagnostics/Sepsis/Models/'
model_name = 'lstm/64_2x96_48_lr1_mom0_5'


#TODO: add more args, including train/test, etc.
#TODO: change which GPU is being used
parser = argparse.ArgumentParser(description='GPU control')
parser.add_argument('--disable-cuda', action='store_true',
                    help='Disable CUDA')
args = parser.parse_args()
args.device = None
if not args.disable_cuda and torch.cuda.is_available():
    args.device = torch.device('cuda')
    torch.set_default_tensor_type('torch.cuda.DoubleTensor')
else:
    args.device = torch.device('cpu')
    torch.set_default_tensor_type('torch.DoubleTensor')

#TODO: control with args
#       be careful since some parameters are model specic!
#       probably just move model to the top and specify all of its args at start

epochs = 20
batch_size = 1024
save_rate = 10
save = False
load_model = False #'/home/osvald/Projects/Diagnostics/Sepsis/Models/lstm/32_2x64_l01_p25/model_epoch80'
load_folder = False #'/home/osvald/Projects/Diagnostics/Sepsis/Models/lstm/32_2x64_l01_p25'
offset = 0

n = 40336
split = 0.8
ind = np.random.permutation(n)
div = int(n * split)
partition = dict([])
partition['train'] = list(ind[:div])
partition['validation'] = list(ind[div:n])


'''TCN'''
#model = TCN(40, 1, [64, 48], fcl=32, kernel_size=2, dropout=0.4).to(args.device)
#criterion = nn.BCEWithLogitsLoss(pos_weight=torch.DoubleTensor([1.8224]).to(args.device), reduction='none')
#optimizer = optim.Adam(model.parameters(), lr=0.0001, betas=(0.75, 0.99))
##optimizer = optim.SGD(model.parameters(), lr=0.0001, momentum=0.9)

model = lstm(embedding=64, hidden_size=64, fcl=32, num_layers=2, 
             batch_size=batch_size, fcl_out=False, embed=True, droprate=0.25).to(args.device)

#model = LSTM_attn(embedding=32, hidden_size=64, num_layers=2, batch_size=batch_size, embed=True, droprate=0.25).to(args.device)
criterion = nn.BCEWithLogitsLoss(pos_weight=torch.DoubleTensor([1.8224]).to(args.device), reduction='none') #1.8224
#criterion = nn.MSELoss(reduction='none') #1.8224

optimizer = optim.SGD(model.parameters(), lr=1, momentum=0.5)

lr_lambda = lambda epoch: 0.9 ** (epoch)
scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
#scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, verbose=True, factor=0.1, patience=5)

if load_model:
    model.load_state_dict(torch.load(load_model))
    train_losses = np.concatenate((np.load(load_folder +'/train_losses.npy'), np.zeros(epochs)))
    train_pos_acc = np.concatenate((np.load(load_folder +'/train_pos_acc.npy'), np.zeros(epochs)))
    train_neg_acc = np.concatenate((np.load(load_folder +'/train_neg_acc.npy'), np.zeros(epochs)))
    val_losses = np.concatenate((np.load(load_folder +'/val_losses.npy'), np.zeros(epochs)))
    val_pos_acc = np.concatenate((np.load(load_folder +'/val_pos_acc.npy'), np.zeros(epochs)))
    val_neg_acc = np.concatenate((np.load(load_folder +'/val_neg_acc.npy'), np.zeros(epochs)))
    utility = np.concatenate((np.load(load_folder +'/utility.npy'), np.zeros(epochs)))
    p_utility = np.concatenate((np.load(load_folder +'/p_utility.npy'), np.zeros(epochs)))
    plotter(model_name, utility, p_utility, train_losses, train_pos_acc, train_neg_acc,
        val_losses, val_pos_acc, val_neg_acc)

    partition['train'] = np.load(load_folder +'/train.npy')
    partition['validation'] = np.load(load_folder +'/valid.npy')
    print('loaded train data length:', len(partition['train']))
    print('loaded valid data length:', len(partition['validation']))
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
    p_utility = np.zeros(epochs)
    if save: #save train/validation split if model is not being loaded
        np.save(save_path+model_name +'/train', np.array(partition['train']))
        np.save(save_path+model_name +'/valid', np.array(partition['validation']))


train_data = Dataset(partition['train'], train_path, util_weights=True)
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, collate_fn=util_collate_fn)

val_data = Dataset(partition['validation'], train_path, util_weights=True)
val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=True, collate_fn=util_collate_fn)


def lstm_train():
    running_loss, pos_total, pos_correct, neg_total, neg_correct  = 0, 0, 0, 0, 0
    for batch, labels, seq_len in train_loader:
        # pass to GPU if available
        batch, labels = batch.to(args.device), labels.to(args.device)
        max_len = labels.shape[1]

        optimizer.zero_grad()
        outputs = model(batch, seq_len, max_len, batch_size)
        outputs = outputs.view(-1, max_len)
        loss = criterion(outputs, labels)
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
    
    train_losses[epoch] = running_loss/len(train_loader)
    train_pos_acc[epoch] = pos_correct/pos_total
    train_neg_acc[epoch] = neg_correct/neg_total

def lstm_test():
    predictions, truth = [], []
    proc_pred = []
    running_loss, pos_total, pos_correct, neg_total, neg_correct  = 0, 0, 0, 0, 0
    with torch.set_grad_enabled(False):
        for batch, labels, seq_len in val_loader:
            # pass to GPU if available
            batch, labels = batch.to(args.device), labels.to(args.device)
            max_len = labels.shape[1]

            outputs = model(batch, seq_len, max_len, batch_size)
            outputs = outputs.view(-1, max_len)
            loss = criterion(outputs, labels)
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
                #post-process test, still needs to detect non-sepsis patients
                best = [len(prediction), len(prediction)-1] # [err, idx]
                for j in range(1, len(prediction)+1):
                    err = float(abs((prediction - (np.concatenate(( np.zeros(j) , np.ones(len(prediction) - (j)) ))))).sum())
                    if err < best[0]:
                        best = [err, j]
                proc = np.concatenate(( np.zeros(best[1]) , np.ones(len(prediction) - (best[1]) )))
                proc_pred.append(proc)
                predictions.append(prediction)
                truth.append(targets)

        val_losses[epoch] = running_loss/len(val_loader)
        val_pos_acc[epoch] = pos_correct/pos_total
        val_neg_acc[epoch] = neg_correct/neg_total
        utility[epoch] = evaluate_sepsis_score(truth, predictions)
        p_utility[epoch] = evaluate_sepsis_score(truth, proc_pred)


def lstm_util_train(util_enable=True):
    running_loss, pos_total, pos_correct, neg_total, neg_correct  = 0, 0, 0, 0, 0
    for batch, labels, util, seq_len in train_loader:
        # pass to GPU if available
        batch, labels, util = batch.to(args.device), labels.to(args.device), util.to(args.device)
        max_len = labels.shape[1]

        optimizer.zero_grad()
        outputs = model(batch, seq_len, max_len)
        outputs = outputs.view(-1, max_len)

        if util_enable:
            loss = torch.mean(criterion(outputs, labels) * util)
        else:
            loss = torch.mean(criterion(outputs, labels))
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

def lstm_util_test(util_enable=True):
    predictions, truth = [], []
    proc_pred = []
    running_loss, pos_total, pos_correct, neg_total, neg_correct  = 0, 0, 0, 0, 0
    with torch.set_grad_enabled(False):
        for batch, labels, util, seq_len in val_loader:
            # pass to GPU if available
            batch, labels = batch.to(args.device), labels.to(args.device)
            max_len = labels.shape[1]

            outputs = model(batch, seq_len, max_len, batch_size)
            outputs = outputs.view(-1, max_len)
            if util_enable:
                loss = torch.mean(criterion(outputs, labels) * util)
            else:
                loss = torch.mean(criterion(outputs, labels))
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
                #post-process test, still needs to detect non-sepsis patients
                best = [len(prediction), len(prediction)-1] # [err, idx]
                for j in range(1, len(prediction)+1):
                    err = float(abs((prediction - (np.concatenate(( np.zeros(j) , np.ones(len(prediction) - (j)) ))))).sum())
                    if err < best[0]:
                        best = [err, j]
                proc = np.concatenate(( np.zeros(best[1]) , np.ones(len(prediction) - (best[1]) )))
                proc_pred.append(proc)
                predictions.append(prediction)
                if targets.sum() > 0: #TODO: make condition dependent on "util optim" arg
                    onset = np.argwhere(targets == 1)[0][0] 
                    targets[onset:onset+6] = 0 # remove label extension
                truth.append(targets)

        val_losses[epoch] = running_loss/len(val_loader) * 100 # scaling applied at end for ease of reading graph
        val_pos_acc[epoch] = pos_correct/pos_total
        val_neg_acc[epoch] = neg_correct/neg_total
        utility[epoch] = evaluate_sepsis_score(truth, predictions)
        p_utility[epoch] = evaluate_sepsis_score(truth, proc_pred)


def tcn_train():
    running_loss, pos_total, pos_correct, neg_total, neg_correct  = 0, 0, 0, 0, 0
    for batch, labels, util, seq_len in train_loader:
        # pass to GPU if available
        batch, labels = batch.to(args.device), labels.to(args.device)
        outputs = model(batch.permute(0,2,1))
        loss = torch.mean(criterion(outputs, labels) * util)
        loss.backward()
        optimizer.step()
        running_loss += loss.cpu().data.numpy()

        # Train Accuracy
        for i in range(labels.shape[0]):
            targets = labels.data[i,:int(seq_len[i])].cpu().numpy()
            predictions = torch.round(torch.sigmoid(outputs.data[i, :int(seq_len[i])])).cpu().numpy()
            match = targets == predictions
            pos_total += (targets == 1).sum()
            neg_total += (targets == 0).sum()
            pos_correct += (match * (targets == 1)).sum()
            neg_correct += (match * (targets == 0)).sum()

    train_losses[epoch] = running_loss/len(train_loader) * 100 # scaling factor
    train_pos_acc[epoch] = pos_correct/pos_total
    train_neg_acc[epoch] = neg_correct/neg_total

def tcn_test():
    predictions, truth = [], []
    proc_pred = []
    running_loss, pos_total, pos_correct, neg_total, neg_correct  = 0, 0, 0, 0, 0
    with torch.set_grad_enabled(False):
        for batch, labels, util, seq_len in val_loader:
            #labels = labels[:,-1].view(-1,1) # if single output
            batch, labels = batch.to(args.device), labels.to(args.device)
            outputs = model(batch.permute(0,2,1))
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
                #post-process test, still needs to detect non-sepsis patients
                best = [len(prediction), len(prediction)-1] # [err, idx]
                for j in range(1, len(prediction)+1):
                    err = float(abs((prediction - (np.concatenate(( np.zeros(j) , np.ones(len(prediction) - (j)) ))))).sum())
                    if err < best[0]:
                        best = [err, j]
                proc = np.concatenate(( np.zeros(best[1]) , np.ones(len(prediction) - (best[1]) )))
                proc_pred.append(proc)
                predictions.append(prediction)
                truth.append(targets)

        val_losses[epoch] = running_loss/len(val_loader) * 100 # scaling applied at end for ease of reading graph
        val_pos_acc[epoch] = pos_correct/pos_total
        val_neg_acc[epoch] = neg_correct/neg_total
        utility[epoch] = evaluate_sepsis_score(truth, predictions)
        p_utility[epoch] = evaluate_sepsis_score(truth, proc_pred)


'''__main__'''
start = time.time()
for epoch in range(offset, offset + epochs):
    model.train()
    lstm_util_train()
    model.eval()
    lstm_util_test()
    #scheduler.step(train_losses[epoch]) 
    scheduler.step() 

    show_prog(epoch, utility[epoch], p_utility[epoch], train_losses[epoch], val_losses[epoch], train_pos_acc[epoch],
              train_neg_acc[epoch], val_pos_acc[epoch], val_neg_acc[epoch], (time.time() - start))
    
    if save:
        save_prog(model, save_path+model_name, utility, p_utility, train_losses, val_losses, train_pos_acc,
              train_neg_acc, val_pos_acc, val_neg_acc, epoch, save_rate)
    else: print('MODEL AND DATA ARE NOT BEING SAVED')

# PLOT GRAPHS
plotter(model_name, utility, p_utility, train_losses, train_pos_acc, train_neg_acc,
        val_losses, val_pos_acc, val_neg_acc)