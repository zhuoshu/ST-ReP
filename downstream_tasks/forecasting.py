import torch
import torch.nn as nn
import numpy as np
import time

from einops import rearrange, repeat
from torch.utils.data import TensorDataset, DataLoader, Dataset
from utils.tools import cal_metric_log

import os
import gc

import sklearn
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split


## This fit_ridge function is derived from and modified based on TS2Vec [https://github.com/zhihanyue/ts2vec].
## We treat STS as MTS for the training of ridge regression, following the evaluation setup of previous works.
## And we predcit the futrue series of whole STS variables later.
def fit_ridge(logger, train_features, train_y, valid_features, valid_y, MAX_SAMPLES=100000):
    # If the training set is too large, subsample MAX_SAMPLES examples
    logger.info(f'before split: train_features.shape ={train_features.shape}')
    logger.info(f'before split: train_y.shape = {train_y.shape}')
    logger.info(f'before split: valid_features.shape = {valid_features.shape}')
    logger.info(f'before split: valid_y.shape = {valid_y.shape}')

    if train_features.shape[0] > MAX_SAMPLES:
        split = train_test_split(
            train_features, train_y,
            train_size = MAX_SAMPLES, random_state=0
        )
        train_features = split[0]
        train_y = split[2]
    else: 
        train_features, train_y = sklearn.utils.shuffle(train_features, train_y, random_state=0)

    if valid_features.shape[0] > MAX_SAMPLES:
        split = train_test_split(
            valid_features, valid_y,
            train_size = MAX_SAMPLES, random_state=0
        )
        valid_features = split[0]
        valid_y = split[2]
    else: 
        valid_features, valid_y = sklearn.utils.shuffle(valid_features, valid_y, random_state=0)


    logger.info(f'after split: train_features.shape ={train_features.shape}')
    logger.info(f'after split: train_y.shape ={train_y.shape}')
    logger.info(f'after split: valid_features.shape ={valid_features.shape}')
    logger.info(f'after split: valid_y.shape ={valid_y.shape}')

    ridge_alphas = [0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000]
    valid_results = []
    for ridge_alpha in ridge_alphas:
        lr = Ridge(alpha=ridge_alpha).fit(train_features, train_y)
        valid_pred = lr.predict(valid_features)
        score = np.sqrt(((valid_pred - valid_y) ** 2).mean()) + np.abs(valid_pred - valid_y).mean()
        valid_results.append(score)
        # logger.info(f'ridge_alpha:{ridge_alpha}, results:{score}')

    best_ridge_alpha = ridge_alphas[np.argmin(valid_results)]
    logger.info(f'best_ridge_alpha:{best_ridge_alpha}')

    lr = Ridge(alpha=best_ridge_alpha)
    lr.fit(train_features, train_y)
    return lr

class simpleDataset(Dataset):
    def __init__(self, data, seq_len) -> None:
        super().__init__()
        self.seq_len = seq_len
        self.data_x = data[0]
        self.data_stamp = data[1]

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len

        seq_x = self.data_x[s_begin:s_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        return seq_x, seq_x_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len + 1


def get_rep(model, data, seq_len, batch_size, device):
    dataset = simpleDataset(data, seq_len=seq_len)
    
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    model.eval()
    with torch.no_grad():
        output = []
        for batch in loader:  
            x = batch[0].float().to(device)   
            x_timestamp = batch[1].int().to(device)
            out, _ = model.get_rep([x, x_timestamp])                       
            output.append(out[:,-1:,:,:].cpu().numpy())
            torch.cuda.empty_cache()
        output = np.concatenate(output, axis=0)        
    return output


def generate_pred_samples_forHL(feature, data, seq_len, pred_len):
    x_length = feature.shape[0] - -seq_len - pred_len + 1 # feature.shape[0] = data.shape[0]
    y_length = data.shape[0] - seq_len - pred_len + 1
    y = []
    for index in range(y_length):
        r_begin = index + seq_len
        r_end = r_begin + pred_len
        y.append(data[r_begin:r_end])

    x = feature[seq_len-1:-pred_len]
    y = np.stack(y, axis=0)
    print('x.shape:',x.shape)
    print('y.shape:',y.shape)
    return x, y

def eval_HL(logger, expid, args, data_list, scaler, ds_pred_len_list, save_path):
    
    input_length = args.ds_input_length
    
    for i, pred_len in enumerate(ds_pred_len_list):
        logger.info(f'############# [{input_length}->{pred_len}]  #############')
        
        test_data = data_list[0]
        t1 = time.time()
        testx, realy = generate_pred_samples_forHL(test_data[0], test_data[0], seq_len=args.ds_input_length, pred_len=pred_len)
        yhat = np.expand_dims(testx, axis = 1)
        yhat = np.repeat(yhat, repeats=pred_len, axis=1)
        logger.info(f'yhat.shape:{yhat.shape}')
        logger.info(f'realy.shape:{realy.shape}')
        inference_time = time.time()-t1
        logger.info(f'inference_time:{inference_time}')

        logger.info('Norm Metric calculation and log:')
        if not os.path.exists(f'./EXP_results/{args.modelid}/'):
            os.makedirs(f'./EXP_results/{args.modelid}/')
        result_norm_csv_path = f'./EXP_results/{args.modelid}/{args.modelid}_{args.dataset_name}_Results_norm.csv'
        
        cal_metric_log(logger, expid, args, yhat, realy, result_norm_csv_path, save_path=save_path, flag='np')

        ######### manually free memory #########
        t = time.time()
        del yhat, realy
        gc.collect()
        gc_time = time.time() - t
        # logger.info(f'gc_time: {gc_time}')
        logger.info(f'############# [{input_length}->{pred_len}] finished #############')


def get_ridge_dataset(data, seq_len, pred_len):
    length = data.shape[0] - seq_len - pred_len + 1
    x = []
    y = []
    for index in range(length):
        s_begin = index
        s_end = s_begin + seq_len
        r_begin = s_end 
        r_end = r_begin + pred_len
        x.append(data[s_begin:s_end])
        y.append(data[r_begin:r_end])

    x = np.stack(x, axis=0)
    y = np.stack(y, axis=0)
    print('x.shape:',x.shape)
    print('y.shape:',y.shape)
    return x, y

def generate_pred_samples_t(feature, data, seq_len, pred_len):
    x_length = feature.shape[0] - pred_len # feature.shape[0] = data.shape[0] - seq_len + 1
    y_length = data.shape[0] - seq_len - pred_len + 1
    y = []
    for index in range(y_length):
        r_begin = index + seq_len
        r_end = r_begin + pred_len
        y.append(data[r_begin:r_end])

    x = feature[:-pred_len]
    y = np.stack(y, axis=0)
    print('x.shape:',x.shape)
    print('y.shape:',y.shape)
    return x, y

def eval_forecasting_ts(logger, expid, args, model, data_list, device, scaler, predict_length_list, save_path):
    train_data, val_data, test_data = data_list    
    logger.info(f'train_data.shape: {train_data[0].shape}, {train_data[1].shape}')
    logger.info(f'test_data.shape: {test_data[0].shape}, {test_data[1].shape}')
    
    useRep = bool(args.useRep)
    if args.modelid == 'Ridge': useRep = False
    input_length = args.ds_input_length

    if useRep:
        infer_batch_size = 32
        t = time.time()
        all_train_repr = get_rep(model, [train_data[0],train_data[1]], seq_len=input_length, batch_size=infer_batch_size,device=device)
        all_val_repr = get_rep(model, [val_data[0],val_data[1]], seq_len=input_length, batch_size=infer_batch_size, device=device)
        all_test_repr = get_rep(model, [test_data[0],test_data[1]], seq_len=input_length, batch_size=infer_batch_size, device=device)
        rep_infer_time = time.time() - t
        logger.info(f'rep_infer_time:{rep_infer_time}')
        logger.info(f'all_train_repr.shape:{all_train_repr.shape}') #n 1 N d
    else:
        all_train_repr, all_val_repr, all_test_repr = train_data[0], val_data[0], test_data[0]

        
    for predict_length in predict_length_list:
        logger.info(f'############# [{input_length}->{predict_length}]  #############')
        if useRep:
            train_repr, train_labels = generate_pred_samples_t(all_train_repr, train_data[0], seq_len=input_length, pred_len=predict_length)
            val_repr, val_labels = generate_pred_samples_t(all_val_repr, val_data[0], seq_len=input_length, pred_len=predict_length)
            test_repr, test_labels = generate_pred_samples_t(all_test_repr, test_data[0], seq_len=input_length, pred_len=predict_length)
        else:
            train_repr, train_labels = get_ridge_dataset(all_train_repr, input_length, predict_length)
            val_repr, val_labels = get_ridge_dataset(all_val_repr, input_length, predict_length)
            test_repr, test_labels = get_ridge_dataset(all_test_repr, input_length, predict_length)
        
        logger.info(f'repr: train:{train_repr.shape}, val:{val_repr.shape}, test:{test_repr.shape}')
        logger.info(f'labels: train:{train_labels.shape}, val:{val_labels.shape}, test:{test_labels.shape}')

        num_train_samples = train_labels.shape[0]
        num_val_samples = val_labels.shape[0]
        num_test_samples, T_pre, N, pre_dim = test_labels.shape

        train_repr = rearrange(train_repr, 'n t N d -> (n N) (t d)')            
        val_repr = rearrange(val_repr, 'n t N d -> (n N) (t d)')
        test_repr = rearrange(test_repr, 'n t N d -> (n N) (t d)')
        test_labels = rearrange(test_labels, 'n T N c -> (n N) (T c)')
        train_labels = rearrange(train_labels, 'n T N c -> (n N) (T c)')
        val_labels = rearrange(val_labels, 'n T N c -> (n N) (T c)')

        logger.info(f'Reshape repr: train:{train_repr.shape}, val:{val_repr.shape}, test:{test_repr.shape}')
        logger.info(f'Reshape labels: train:{train_labels.shape}, val:{val_labels.shape}, test:{test_labels.shape}')

        t = time.time()
        logger.info(f'[{input_length}->{predict_length}] Start to train ridgeregression....')
        lr = fit_ridge(logger, train_repr, train_labels, val_repr, val_labels)
        lr_train_time = time.time() - t
        logger.info(f'lr_train_time: {lr_train_time}')
        
        t = time.time()
        del train_repr, train_labels, val_repr, val_labels
        gc.collect()
        gc_time = time.time() - t
        # logger.info(f'gc_time: {gc_time}')

        t = time.time()
        logger.info(f'[{input_length}->{predict_length}] Start to predict....')
        test_pred = lr.predict(test_repr)
        lr_infer_time = time.time() - t
        logger.info(f'lr_infer_time: {lr_infer_time}')

        test_pred = rearrange(test_pred, '(n N) (T c) -> n T N c', n=num_test_samples, T=predict_length, N=N, c=pre_dim)
        test_labels = rearrange(test_labels, '(n N) (T c) -> n T N c', n=num_test_samples, T=predict_length, N=N, c=pre_dim)

        ######### cal metric and log #########
        logger.info('Norm Metric calculation and log:')
        if not os.path.exists(f'./EXP_results/{args.modelid}/'):
            os.makedirs(f'./EXP_results/{args.modelid}/')
        result_norm_csv_path = f'./EXP_results/{args.modelid}/{args.modelid}_{args.dataset_name}_Results_norm.csv'
        
        cal_metric_log(logger, expid, args, test_pred, test_labels, result_norm_csv_path, save_path, flag='np')    
        
        ######### manually free memory ###
        t = time.time()
        del test_pred, test_labels
        gc.collect()
        gc_time = time.time() - t
        # logger.info(f'gc_time: {gc_time}')

        logger.info(f'############# [{input_length}->{predict_length}] finished #############')

    return
