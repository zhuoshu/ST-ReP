import torch
import numpy as np
import argparse
import time

import csv
from torch import nn
import torch.optim as optim
from models import STReP
from downstream_tasks import forecasting

import os
import torch.nn.functional as F
import sys
from functools import partial
from einops import rearrange

import logging
from logging import getLogger
import pickle
import yaml


from data_provider.data_factory import data_provider
from utils import metrics, scheduler, tools


def get_logger(log_dir, log_filename, name=None):
    logfilepath = os.path.join(log_dir, log_filename)

    logger = logging.getLogger(name)
    
    log_level = 'INFO'
    if log_level.lower() == 'info':
        level = logging.INFO
    elif log_level.lower() == 'debug':
        level = logging.DEBUG
    elif log_level.lower() == 'error':
        level = logging.ERROR
    elif log_level.lower() == 'warning':
        level = logging.WARNING
    elif log_level.lower() == 'critical':
        level = logging.CRITICAL
    else:
        level = logging.INFO
    logger.setLevel(level)
    
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler(logfilepath)
    file_handler.setFormatter(formatter)

    console_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(console_formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    logger.info('Log directory: %s', log_dir)
    return logger


def get_parameter_number(model):
    total_num = sum(p.numel() for p in model.parameters())
    trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_num, trainable_num

def mv(x, scale=1, pool_type='avg'):
    # print('in mv: x.shape=',x.shape)
    if pool_type == 'avg':
        x = F.avg_pool1d(x, scale, scale)
    elif pool_type == 'max':
        x = F.max_pool1d(x, scale, scale)
    # print('in mv: x_mv.shape:',x.shape)
    return x

def ms_loss(inputx, true, reconstructx, predict, scales=[2,4,8,16]):
    if inputx.shape[1]<=16:
        scales = [2,4,8]
    groundtruth = torch.cat([inputx, true], dim=1)
    recon_pred = torch.cat([reconstructx, predict], dim=1)
    groundtruth = rearrange(groundtruth,'b t n c -> (b n) c t')
    recon_pred = rearrange(recon_pred,'b t n c -> (b n) c t')
    x_scale_outputs = []
    pred_scale_outputs=[]
    for i, scale in enumerate(scales):
        x_scale = mv(groundtruth, scale)
        pred_scale = mv(recon_pred, scale)
        x_scale_outputs.append(x_scale)
        pred_scale_outputs.append(pred_scale)
    x_comb = torch.cat(x_scale_outputs,dim=-1)
    pred_comb = torch.cat(pred_scale_outputs, dim=-1)
    loss = metrics.huber_loss(pred_comb, x_comb)
    return loss

class Exp():
    def __init__(self, expid, scaler, args, device):
        self._logger = getLogger()
        self.args = args
        self.device = device
        self.scaler = scaler
        self.clip = 1
        self.predict_length = args.predict_length
        self.epochs = args.epochs
        self.expid = expid
        self.trainloss_record = []
        self.log_in_train_details = []
        self.bestid=None
        
        if args.modelid in ['Ridge','HL']:
            self.model = None
            
        else:
            self.model = self._build_model()
            
            self.optimizer = optim.AdamW(
                self.model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
            self.loss = self._build_loss_func(loss_type=args.loss_type, delta=args.huber_delta)
            self.lr_scheduler = self._build_lr_scheduler()
        
        self.alpha = args.alpha
        self.beta = args.beta
        assert self.alpha + self.beta < 1
    
    def _build_loss_func(self, loss_type='pred_mask_ms', delta=1.0):
        if loss_type == 'pred_only':
            def loss_func(inputx, reconx, inputy, predy):
                pred_loss = metrics.huber_loss(predy, inputy, delta=delta)
                return pred_loss
            return loss_func
        elif loss_type == 'pred_mask':
            def loss_func(inputx, reconx, inputy, predy):
                pred_loss = metrics.huber_loss(predy, inputy, delta=delta)
                reconx_loss = metrics.huber_loss(reconx, inputx, delta=delta)
                full_loss = (1-self.beta)*pred_loss + self.beta*reconx_loss
                return full_loss
            return loss_func
        elif loss_type == 'mask_only':
            def loss_func(inputx, reconx, inputy, predy):
                reconx_loss = metrics.huber_loss(reconx, inputx, delta=delta)
                return reconx_loss
            return loss_func
        elif loss_type == 'pred_mask_ms':
            def loss_func(inputx, reconx, inputy, predy):
                pred_loss = metrics.huber_loss(predy, inputy, delta=delta)
                reconx_loss = metrics.huber_loss(reconx, inputx, delta=delta)
                msloss = ms_loss(inputx, inputy, reconx, predy)
                full_loss = self.alpha*pred_loss + self.beta*reconx_loss + (1-self.alpha-self.beta)*msloss
                return full_loss
            return loss_func

    def _build_model(self):
        model_dict = {
            'STReP':STReP
        }
        model = model_dict[self.args.modelid].Model(self.args).float()
        
        self.args.note = model._getnote() + '_' + self.args.ds_modelid
        self._logger.info(self.args.note)
        
        for p in model.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        model.to(self.device)
        self._logger.info(model)
        for name, param in model.named_parameters():
            self._logger.info(str(name) + '\t' + str(param.shape) + '\t' +
                              str(param.device) + '\t' + str(param.requires_grad))
        return model

    def load_model_fromfile(self, load_path):
        self.model.load_state_dict(torch.load(load_path))
        self._logger.info(f'model loaded! from {load_path}')

    def _build_lr_scheduler(self):
        self.lr_decay = bool(self.args.lr_decay)
        self.lr_scheduler_type = self.args.lr_scheduler_type
        self.lr_decay_ratio = 0.1
        self.lr_T_max = 30
        self.lr_eta_min = 0
        self.lr_warmup_epoch = 5
        self.lr_warmup_init = 1e-6
        if self.lr_decay:
            self._logger.info('You select `{}` lr_scheduler.'.format(
                self.lr_scheduler_type.lower()))
            if self.lr_scheduler_type.lower() == 'cosinelr':
                lr_scheduler = scheduler.CosineLRScheduler(
                    self.optimizer, t_initial=self.epochs, lr_min=self.lr_eta_min, decay_rate=self.lr_decay_ratio,
                    warmup_t=self.lr_warmup_epoch, warmup_lr_init=self.lr_warmup_init)
            else:
                self._logger.warning('Received unrecognized lr_scheduler, '
                                     'please check the parameter `lr_scheduler`.')
                lr_scheduler = None
        else:
            lr_scheduler = None
        return lr_scheduler

    def _forward(self, input):
        inputx, reconx, inputy, predy = self.model(input)
        return inputx, reconx, inputy, predy

    def _metric(self, predict, real):
        mae = metrics.mae_torch(predict, real).item()
        mse = metrics.mse_torch(predict, real).item()
        mape = metrics.masked_mape_torch(predict, real, mask_val=0.0).item()
        return mae, mape, mse

    def train_batch(self, input, batches_seen):  # input(B,T,N,C), real_val(B,T,N,C)
        self.model.train()
        self.optimizer.zero_grad()
        inputx, reconx, inputy, predy = self._forward(input)
        loss = self.loss(inputx, reconx, inputy, predy)
        loss.backward()
        if self.clip is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)
        self.optimizer.step()
        if self.lr_scheduler is not None:
            if self.lr_scheduler_type.lower() == 'cosinelr':
                self.lr_scheduler.step_update(num_updates=batches_seen)
        if self.args.loss_type == 'mask_only':
            mae, mape, mse = self._metric(reconx, inputx)    
        else: 
            mae, mape, mse = self._metric(predy, inputy)
        return loss.item(), mae, mape, mse

    def eval_batch(self, input):
        self.model.eval()
        inputx, reconx, inputy, predy = self._forward(input)
        loss = self.loss(inputx, reconx, inputy, predy)
        if self.args.loss_type == 'mask_only':
            mae, mape, mse = self._metric(reconx, inputx)    
        else: 
            mae, mape, mse = self._metric(predy, inputy)
        return loss.item(), mae, mape, mse


    def train(self, logger, train_dataloader, val_dataloader, model_save_path):
        # total_para_num, trainable_para_num = get_parameter_number(self.model)   
        all_start_time = time.time()
        his_loss = []
        val_time = []
        train_time = []
        best_validate_loss = np.inf
        validate_score_non_decrease_count = 0
        batches_seen = 0

        for i in range(1, args.epochs + 1):
            train_loss, train_mae, train_mape, train_mse = [], [], [], []
            t1 = time.time()
            for iter, (batch_x, batch_x_mark, batch_y) in enumerate(train_dataloader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.int().to(self.device)
                trainx = [batch_x, batch_x_mark, batch_y]
                scores = self.train_batch(trainx, batches_seen)
                batches_seen += 1
                train_loss.append(scores[0])
                train_mae.append(scores[1])
                train_mape.append(scores[2])
                train_mse.append(scores[3])
                if iter % args.print_every == 0:
                    log = 'Iter: {:03d} [{:d}], Train Loss: {:.4f},Train MAE: {:.4f}, Train MAPE: {:.4f}, Train MSE: {:.4f}'
                    logger.info(log.format(
                        iter, batches_seen, train_loss[-1], train_mae[-1], train_mape[-1], train_mse[-1]))
            t2 = time.time()
            train_time.append(t2 - t1)

            # validation
            valid_loss, valid_mae, valid_mape, valid_mse = [], [], [], []
            s1 = time.time()
            for iter, (batch_x, batch_x_mark, batch_y) in enumerate(val_dataloader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.int().to(self.device)
                valx = [batch_x, batch_x_mark, batch_y]
                scores = self.eval_batch(valx)
                valid_loss.append(scores[0])
                valid_mae.append(scores[1])
                valid_mape.append(scores[2])
                valid_mse.append(scores[3])
            s2 = time.time()
            log = 'Epoch: {:03d}, Inference Time: {:.4f} secs'
            logger.info(log.format(i, (s2 - s1)))
            val_time.append(s2 - s1)

            mtrain_loss = np.mean(train_loss)
            mtrain_mae = np.mean(train_mae)
            mtrain_mape = np.mean(train_mape)
            mtrain_mse = np.mean(train_mse)

            mvalid_loss = np.mean(valid_loss)
            mvalid_mae = np.mean(valid_mae)
            mvalid_mape = np.mean(valid_mape)
            mvalid_mse = np.mean(valid_mse)

            if self.lr_scheduler is not None:
                if self.lr_scheduler_type.lower() == 'reducelronplateau':
                    self.lr_scheduler.step(mvalid_loss)
                elif self.lr_scheduler_type.lower() == 'cosinelr':
                    self.lr_scheduler.step(i)
                else:
                    self.lr_scheduler.step()

            his_loss.append(mvalid_loss)
            self.trainloss_record.append(
                [mtrain_loss, mtrain_mae, mtrain_mape, mtrain_mse, mvalid_loss, mvalid_mae, mvalid_mape, mvalid_mse])
            log = 'Epoch: {:03d}, Train Loss: {:.4f}, Train MAE: {:.4f} Train MAPE: {:.4f}, Train MSE: {:.4f}, '
            log += 'Valid Loss: {:.4f}, Valid MAE: {:.4f}, Valid MAPE: {:.4f}, Valid MSE: {:.4f}, Training Time: {:.4f}/epoch'
            logger.info(log.format(i, mtrain_loss, mtrain_mae, mtrain_mape, mtrain_mse,
                                   mvalid_loss, mvalid_mae, mvalid_mape, mvalid_mse, (t2 - t1)))
            self.log_in_train_details.append([i, mtrain_loss, mtrain_mae, mtrain_mape, mtrain_mse, 
                                              mvalid_loss, mvalid_mae, mvalid_mape, mvalid_mse, (t2 - t1)])

            if best_validate_loss > mvalid_loss:
                best_validate_loss = mvalid_loss
                validate_score_non_decrease_count = 0
                torch.save(self.model.state_dict(), model_save_path + "best.pth")
                logger.info('got best validation result: {:.4f}, {:.4f}, {:.4f}'.format(
                    mvalid_loss, mvalid_mape, mvalid_mse))
            else:
                validate_score_non_decrease_count += 1

            if args.early_stop and validate_score_non_decrease_count >= args.early_stop_step:
                break
        
        self.avg_train_time = np.mean(train_time)
        self.avg_inference_time = np.mean(val_time)
        logger.info("Average Training Time: {:.4f} secs/epoch".format(self.avg_train_time))
        logger.info("Average Inference Time: {:.4f} secs".format(self.avg_inference_time))
        self.training_time = (time.time() - all_start_time) / 60

        self.bestid = np.argmin(his_loss)
        logger.info(f"Training finished! bestid: {self.bestid}")
        logger.info(f"The valid loss on best model is {str(round(his_loss[self.bestid], 4))}")
    


def main(args):
    expid = time.strftime("%m%d%H%M", time.localtime())
    if args.load_expid != 'none':
        expid += f'load{args.load_expid}'
    device = torch.device(args.device)

    if os.path.exists(f'./configurations/{args.config_file}'):
        configs = yaml.load(
            open(f'./configurations/{args.config_file}'), 
            Loader=yaml.FullLoader
        )
        config_file = argparse.Namespace(**configs)

        if config_file.modelid != 'default': args.modelid = config_file.modelid
        if args.input_length == -1: args.input_length = config_file.input_length
        if args.predict_length == -1: args.predict_length = config_file.predict_length
        if args.M == -1: args.M = config_file.M
        if args.d_model == -1: args.d_model = config_file.d_model
        if args.num_layers == -1: args.num_layers = config_file.num_layers
        if args.time_factor == -1: args.time_factor = config_file.time_factor

        args.ds_input_length = args.input_length

        args.root_path = config_file.root_path
        args.data_path = config_file.data_path
        args.dataset_name =config_file.dataset_name
        args.in_dim = config_file.in_dim
        args.label_len = 0
        args.n_heads = config_file.n_heads
        args.tau = config_file.tau
        args.alpha = config_file.alpha
        args.beta = config_file.beta

    assert args.modelid != 'default' and args.dataset_name != 'none'

    
    save_path = f'./EXP_experiments/{args.modelid}/{args.dataset_name}/{args.modelid}_{args.dataset_name}_{args.input_length}to{args.predict_length}_exp' + str(
        expid) + "/"
    log_path = save_path
    log_filename = f'{args.modelid}_{args.dataset_name}_{args.input_length}to{args.predict_length}_exp{expid}.log'
    if not os.path.exists(log_path):
        os.makedirs(log_path)

        
    if args.load_expid == 'none':
        with open(f'{log_path}/config.pkl','wb') as f:
            pickle.dump(args,f)
        with open(f'{log_path}/config.yaml', "w", encoding="utf-8") as f:
            yaml.dump(args,f)
    else:
        exp_path = f'./EXP_experiments/{args.modelid}/{args.dataset_name}/{args.modelid}_{args.dataset_name}_{args.input_length}to{args.predict_length}_exp{args.load_expid}/'
        now_load_expid = args.load_expid
        now_ds_task = args.ds_task
        if os.path.exists(f'{exp_path}/config.pkl'):
            with open(f'{exp_path}/config.pkl', 'rb') as file:
                args = pickle.load(file)
                args.load_expid = now_load_expid
                args.ds_task = now_ds_task
                print('args loaded!')

    dataset_info = {        
        'PEMS04': [307, '5m', 288],
        'PEMS08': [170, '5m', 288],
        'ca':[8600,'15m',96],
        'humidity2016': [2048,'1h',24],
        'temperature2016': [2048,'1h',24],
        'SDWPF':[134,'10m',144]
    }
    
    ds_pred_len_info ={
        'PEMS04':[12,24,48,96],
        'PEMS08':[12,24,48,96],
        'temperature2016':[12,24,48,96],
        'humidity2016':[12,24,48,96],
        'ca':[4,8,12,16],
        'SDWPF':[12,24,48,96]
    }
    args.num_nodes, args.freq, args.slice_size_per_day = dataset_info[args.dataset_name]
    args.ds_pred_len_list = ds_pred_len_info[args.dataset_name]


    logger = get_logger(log_path, log_filename)
    exp_name = f'{args.modelid}_{args.dataset_name}'
    logger.info(exp_name)
    
    train_dataset, train_dataloader = data_provider(args, flag='train')
    val_dataset, val_dataloader = data_provider(args, flag='val')
    test_dataset, test_dataloader = data_provider(args, flag='test')

    scaler = train_dataset.scaler
    
    if args.input_length == args.ds_input_length:
        data_list =[
            [train_dataset.data_x, train_dataset.data_stamp],
            [val_dataset.data_x, val_dataset.data_stamp],
            [test_dataset.data_x, test_dataset.data_stamp]
        ]
        ds_scaler = train_dataset.scaler
    else:
        ds_train_dataset, _ = data_provider(args, flag='train', forDownstream=True)
        ds_val_dataset, _ = data_provider(args, flag='val', forDownstream=True)
        ds_test_dataset, _ = data_provider(args, flag='test', forDownstream=True)
        ds_scaler = ds_train_dataset.scaler
        data_list=[
                [ds_train_dataset.data_x, ds_train_dataset.data_stamp], 
                [ds_val_dataset.data_x, ds_val_dataset.data_stamp],
                [ds_test_dataset.data_x, ds_test_dataset.data_stamp]
            ]


    logger.info(args)    

    args.pre_dim = args.in_dim
    engine = Exp(expid=expid,scaler=scaler, args=args, device=device)

    if args.modelid in ['Ridge','HL'] or bool(args.useRep)==False:
        pass
    else:
        if args.load_expid != 'none':
            load_path = f'./EXP_experiments/{args.modelid}/{args.dataset_name}/{args.modelid}_{args.dataset_name}_{args.input_length}to{args.predict_length}_exp{args.load_expid}/best.pth'
            engine.load_model_fromfile(load_path=load_path)
        else:
            logger.info('start training:')
            engine.train(logger=logger,train_dataloader=train_dataloader, val_dataloader=val_dataloader,model_save_path=save_path)
        
    logger.info('start downstream task:')
    if args.modelid == 'HL':
        forecasting.eval_HL(logger=logger, expid=expid, args=args, data_list=data_list, scaler=ds_scaler, 
                             ds_pred_len_list=args.ds_pred_len_list, save_path=save_path)    
    elif args.ds_modelid=='ridge':
        forecasting.eval_forecasting_ts(logger=logger, expid=expid, args=args, model=engine.model, data_list=data_list, 
                                                device=device, scaler=ds_scaler, predict_length_list=args.ds_pred_len_list,
                                            save_path=save_path)
    else:
        logger.info(f'there is no ds_modelid named {args.ds_modelid}')

if __name__ == "__main__":
    t1 = time.time()
    parser = argparse.ArgumentParser()

    # load configuration from files
    parser.add_argument('--config_file', type=str, default='none')

    # the information about the raw data preprocessing
    parser.add_argument('--dataset_name', type=str, default='none')
    parser.add_argument('--input_length', type=int, default=-1)
    parser.add_argument('--predict_length', type=int, default=-1)
    parser.add_argument('--mask_ratio', type=float, default=0.25)
    parser.add_argument('--num_workers', type=int, default=10)

    # the hyper-parameter-setting in the model
    parser.add_argument('--modelid', type=str, default='STReP')
    parser.add_argument('--d_model', type=int, default=-1) # d in the paper
    parser.add_argument('--n_heads', type=int, default=-1) # the number of heads
    parser.add_argument('--M', type=int, default=-1) # m in the paper, the size of proxy tensor in Spatail Extracion 
    parser.add_argument('--time_factor', type=int, default=-1) # p in the paper, the size of compression space 
    parser.add_argument('--num_layers', type=int, default=-1) # L, the number of encoder layers
    parser.add_argument('--hasCross', type=int, default=1) # use cross-time concatenation in the ST-Embedding module 
    parser.add_argument('--tau', type=int, default=3) # the convolution kernel size in Projecter2 
    parser.add_argument('--spatial_dropout', type=float, default=0.1)
    parser.add_argument('--spatial_att_dropout', type=float, default=0.1)
    parser.add_argument('--st_emb_dropout', type=float, default=0.1)
    parser.add_argument('--rep_dropout', type=float, default=0.1)

    # for the downstream tasks
    parser.add_argument('--ds_modelid', type=str, default='ridge')

    # the hyper-parameter-setting of the training process
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--learning_rate', type=float,default=0.001)
    parser.add_argument('--weight_decay', type=float,default=0.0001)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--print_every', type=int, default=50)
    parser.add_argument('--early_stop', type=bool, default=True)
    parser.add_argument('--early_stop_step', type=int, default=10)
    parser.add_argument('--lr_decay', type=int, default=1)
    parser.add_argument('--huber_delta', type=int, default=2, help='delta in huber loss')
    parser.add_argument('--lr_scheduler_type', type=str, default='cosinelr')
    parser.add_argument('--save_output', type=int, default=0)

    parser.add_argument('--loss_type', type=str, default='pred_mask_ms')
    parser.add_argument('--alpha', type=float, default=0.2)
    parser.add_argument('--beta', type=float, default=0.5)
    parser.add_argument('--useRep', type=int, default=1)

    parser.add_argument('--load_expid', type=str, default='none')
    parser.add_argument('--note', type=str, default='ST-ReP')
    args = parser.parse_args()

    main(args)
    t2 = time.time()
    print("Total time spent: {:.4f}".format(t2 - t1))
    