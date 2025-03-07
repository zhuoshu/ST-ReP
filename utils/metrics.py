import numpy as np
import torch

def huber_loss(preds, labels, delta=1.0):
    residual = torch.abs(preds - labels)
    condition = torch.le(residual, delta)
    small_res = 0.5 * torch.square(residual)
    large_res = delta * residual - 0.5 * delta * delta
    return torch.mean(torch.where(condition, small_res, large_res))


def masked_huber_loss(preds, labels, delta=1.0, null_val=np.nan):
    labels[torch.abs(labels) < 1e-4] = 0
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = labels.ne(null_val)
    mask = mask.float()
    mask /= torch.mean(mask)
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    residual = torch.abs(preds - labels)
    condition = torch.le(residual, delta)
    small_res = 0.5 * torch.square(residual)
    large_res = delta * residual - 0.5 * delta * delta
    loss = torch.where(condition, small_res, large_res)
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)


def mae_torch(preds,labels):
    loss = torch.mean(torch.abs(preds - labels))
    return loss
def mse_torch(preds,labels):
    loss = torch.mean(torch.square(preds - labels))
    return loss
def rmse_torch(preds,labels):
    loss = torch.mean(torch.square(preds - labels))
    return loss**0.5


def masked_mae_torch(preds, labels, null_val=np.nan, mask_val=np.nan):
    # print('mask_val:',mask_val)
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = labels.ne(null_val)
    if mask_val != None:
        mask &= torch.gt(labels, mask_val)
        preds = torch.masked_select(preds, mask)
        labels = torch.masked_select(labels, mask)
    return torch.mean(torch.abs(labels-preds))


def masked_mape_torch(preds, labels, null_val=np.nan, mask_val=0):
    labels[torch.abs(labels) < 1e-4] = 0
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = labels.ne(null_val)
    if mask_val != None:
        mask &= torch.gt(labels, mask_val)
        preds = torch.masked_select(preds, mask)
        labels = torch.masked_select(labels, mask)
    return torch.mean(torch.abs(torch.div((labels - preds), labels)))


def masked_mse_torch(preds, labels, null_val=np.nan, mask_val=np.nan):
    labels[torch.abs(labels) < 1e-4] = 0
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = labels.ne(null_val)
    if mask_val != None:
        mask &= torch.gt(labels, mask_val)
        preds = torch.masked_select(preds, mask)
        labels = torch.masked_select(labels, mask)
    return torch.mean(torch.square(labels-preds))



def masked_rmse_torch(preds, labels, null_val=np.nan, mask_val=np.nan):
    return torch.sqrt(masked_mse_torch(preds=preds, labels=labels,
                                       null_val=null_val, mask_val=mask_val))

def metric_torch(predict, real, mask_val=-1):
    if mask_val == -1:
        nmae = mae_torch(predict, real).item()
        nmse = mse_torch(predict,real).item()
        nrmse = nmse ** 0.5
        mape = masked_mape_torch(predict, real, mask_val=0.0).item()*100
    else:
        nmae = masked_mae_torch(predict, real, mask_val=mask_val).item()
        nmse = masked_mse_torch(predict, real, mask_val=mask_val).item()
        nrmse = nmse ** 0.5
        mape = masked_mape_torch(predict, real, mask_val=mask_val).item()*100
    return nmae, mape, nmse, nrmse

def mae_np(preds,labels):
    loss = np.mean(np.abs(preds - labels))
    return loss
def mse_np(preds,labels):
    loss = np.mean(np.square(preds - labels))
    return loss
def rmse_np(preds,labels):
    loss = np.mean(np.square(preds - labels))
    return loss**0.5



def masked_mape_np(y_pred, y_true, null_val=np.nan, mask_val=0):
    y_true[np.abs(y_true) < 1e-4] = 0
    if np.isnan(null_val):
            mask = ~np.isnan(y_true)
    else:
        mask = np.not_equal(y_true, null_val)
    if mask_val != None:
        mask &= np.where(y_true > (mask_val), True, False)
        y_true = y_true[mask]
        y_pred = y_pred[mask]
    return np.mean(np.absolute(np.divide((y_true - y_pred), y_true)))

def masked_mae_np(y_pred, y_true, null_val=np.nan, mask_val=np.nan):
    y_true[np.abs(y_true) < 1e-4] = 0
    if np.isnan(null_val):
        mask = ~np.isnan(y_true)
    else:
        mask = np.not_equal(y_true, null_val)
    if mask_val != None:
        mask &= np.where(y_true > (mask_val), True, False)
        y_true = y_true[mask]
        y_pred = y_pred[mask]
    return np.mean(np.absolute(y_pred-y_true))

def masked_mse_np(y_pred, y_true, null_val=np.nan, mask_val=np.nan):
    y_true[np.abs(y_true) < 1e-4] = 0
    if np.isnan(null_val):
        mask = ~np.isnan(y_true)
    else:
        mask = np.not_equal(y_true, null_val)
    if mask_val != None:
        mask &= np.where(y_true > (mask_val), True, False)
        y_true = y_true[mask]
        y_pred = y_pred[mask]
    return np.mean(np.square(y_pred-y_true))

def masked_rmse_np(y_pred, y_true, null_val=np.nan, mask_val=np.nan):
    return np.sqrt(masked_mse_np(y_pred=y_pred, y_true=y_true,
                                       null_val=null_val, mask_val=mask_val))


def metric_np(predict, real, mask_val=-1):
    if mask_val == -1:
        nmae = mae_np(predict, real).item()
        nmse = mse_np(predict, real).item()
        nrmse = nmse ** 0.5
        mape = masked_mape_np(predict, real, mask_val=0.0).item()*100
    else:
        nmae = masked_mae_np(predict, real,  mask_val=mask_val).item()
        nmse = masked_mse_np(predict, real, mask_val=mask_val).item()
        nrmse = nmse ** 0.5
        mape = masked_mape_np(predict, real, mask_val=mask_val).item()*100
    return nmae, mape, nmse, nrmse



def RSE(pred, true):
    return np.sqrt(np.sum((true - pred) ** 2)) / np.sqrt(np.sum((true - true.mean()) ** 2))


def CORR(pred, true):
    u = ((true - true.mean(0)) * (pred - pred.mean(0))).sum(0)
    d = np.sqrt(((true - true.mean(0)) ** 2 * (pred - pred.mean(0)) ** 2).sum(0))
    return (u / d).mean(-1)


def MAE(pred, true):
    return np.mean(np.abs(pred - true))


def MSE(pred, true):
    return np.mean((pred - true) ** 2)


def RMSE(pred, true):
    return np.sqrt(MSE(pred, true))


def MAPE(pred, true):
    return np.mean(np.abs((pred - true) / true))


def MSPE(pred, true):
    return np.mean(np.square((pred - true) / true))


def metric_my(pred, true):
    mae = MAE(pred, true)
    mse = MSE(pred, true)
    rmse = RMSE(pred, true)
    mape = MAPE(pred, true)
    mspe = MSPE(pred, true)
    rse = RSE(pred, true)
    corr = CORR(pred,true)

    return mae, mse, rmse, mape, mspe,rse,corr
