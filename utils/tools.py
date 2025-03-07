import numpy as np
import torch
import csv
from utils import metrics


def cal_metric_log(logger, expid, args, yhat, realy, result_csv_path, save_path, flag='torch'):
        pred_len = realy.shape[1]
        MAE_list, MAPE_list, MSE_list, RMSE_list = [], [], [], []

        for feature_idx in range(args.pre_dim):
            pred = yhat[..., feature_idx]
            real = realy[..., feature_idx]
       
            metric_mae_func ={'torch':metrics.mae_torch,'np':metrics.mae_np}[flag]
            metric_mape_func ={'torch':metrics.masked_mape_torch,'np':metrics.masked_mape_np}[flag]
            metric_mse_func ={'torch':metrics.mse_torch,'np':metrics.mse_np}[flag]
            MAE = metric_mae_func(pred, real)
            MAPE = metric_mape_func(pred, real, mask_val=0)
            MSE = metric_mse_func(pred, real)
            if flag == 'torch':
                MAE = MAE.item()
                MAPE = MAPE.item()
                MSE = MSE.item()
            RMSE = MSE ** 0.5

            log = '[dim{:d}] On average over {:d} horizons, Test MAE: {:.4f}, Test MAPE: {:.4f}, Test MSE: {:.4f}, Test RMSE: {:.4f}'
            logger.info(log.format(feature_idx, pred_len, MAE, MAPE, MSE, RMSE))

            ## final score 
            MAE_list.append(MAE)
            MAPE_list.append(MAPE)
            MSE_list.append(MSE)
            RMSE_list.append(RMSE)


        with open(result_csv_path, 'a+', newline='')as f0:
            f_csv = csv.writer(f0)
            row = [expid, args.dataset_name, args.note+f'_{pred_len}', args.d_model, pred_len, 'test']
            for feature_idx in range(args.pre_dim):
                row.extend([MSE_list[feature_idx], MAE_list[feature_idx]])
            f_csv.writerow(row)


        logger.info('log and results saved.')

        if bool(args.save_output):
            if flag == 'torch':
                output_path = f'{save_path}/{args.modelid}_{args.dataset_name}_exp{expid}_output.npz'
                np.savez_compressed(output_path, prediction = yhat.cpu().numpy(), truth=realy.cpu().numpy())
                logger.info(f'{output_path}:output npz saved.')
            else:
                output_path = f'{save_path}/{args.modelid}_{args.dataset_name}_exp{expid}_output.npz'
                np.savez_compressed(output_path, prediction = yhat, truth=realy)
                logger.info(f'{output_path}:output npz saved.')