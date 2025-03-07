import os
import numpy as np
import pandas as pd
import os
from torch.utils.data import Dataset, DataLoader
from utils.timefeatures import generate_datetime_series
import warnings
warnings.filterwarnings('ignore')

class StandardScaler():
    """
    Standard the input
    """

    def __init__(self, mean=None, std=None, channel_wise=False):
        self.mean = mean
        self.std = std
        self.channel_wise = channel_wise
        assert channel_wise == False # The normalization in our paper is applied to all variables collectively, rather than individually.

    def fit(self, data):
        if self.channel_wise:
            self.mean = data.mean(axis=0)
            self.std = data.std(axis=0)            
        else:
            self.mean = data.mean()
            self.std = data.std()
        return self

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean

class Dataset_CA(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 data_path='ca_his_2019.h5', 
                 start_time_str='',
                 freq='15min'):
        # size [seq_len, label_len, pred_len]
        self.seq_len = size[0]
        self.label_len = size[1]
        self.pred_len = size[2]
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.start_time_str = start_time_str
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path

        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        data_file = os.path.join(self.root_path, self.data_path)
        df = pd.read_hdf(data_file)
        data = np.expand_dims(df.values, axis=-1)
        print('data.shape:',data.shape)

        train_ratio = 0.6
        test_ratio = 0.2 
        num_train = int(data.shape[0] * train_ratio)
        num_test = int(data.shape[0] * test_ratio)
        num_vali = data.shape[0] - num_train - num_test

        border1s = [0, num_train - self.seq_len, data.shape[0] - num_test - self.seq_len]
        border2s = [num_train, num_train + num_vali, data.shape[0]]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]
        self.slice = slice(border1, border2)
        
        df_stamp = generate_datetime_series(self.start_time_str, self.freq, data.shape[0], 'date')
        df_stamp = df_stamp[['date']][border1:border2]

        t_of_d, d_of_w = None, None
        time_ind = (df_stamp['date'].values - df_stamp['date'].values.astype('datetime64[D]')) / pd.to_timedelta(self.freq)
        t_of_d = time_ind.reshape([time_ind.shape[0],1])
        print('t_of_d.shape',t_of_d.shape)
        
        dow = df_stamp.date.apply(lambda row:row.dayofweek,1)
        d_of_w = dow.values.reshape([dow.shape[0],1])
        print('d_of_w.shape', d_of_w.shape)

        data_stamp = np.concatenate([t_of_d, d_of_w],axis=-1)
        data_stamp = data_stamp.astype(np.int64)
        print('data_stamp.shape:',data_stamp.shape)

        self.scaler.fit(data[border1s[0]:border2s[0]])

        data = self.scaler.transform(data)
        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        
        self.data_stamp = data_stamp
        print(f'data.shape:',self.data_x.shape, self.data_y.shape)
        print(f'data_stamp.shape:',self.data_stamp.shape)

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        # seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_x_mark, seq_y

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)

class Dataset_npz(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                  data_path='PEMS08.npz', 
                  start_time_str='', 
                  freq='h'):
        # size [seq_len, label_len, pred_len]       # info
        self.seq_len = size[0]
        self.label_len = size[1]
        self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        self.flag = flag
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.start_time_str = start_time_str
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        data_file = os.path.join(self.root_path, self.data_path)
        data = np.load(data_file, allow_pickle=True)
        data = data['data'][:, :, 0:1] 
        print('data.shape:',data.shape)

        train_ratio = 0.6
        test_ratio = 0.2       
        num_train = int(data.shape[0] * train_ratio)
        num_test = int(data.shape[0] * test_ratio)
        num_vali = data.shape[0] - num_train - num_test

        border1s = [0, num_train - self.seq_len, data.shape[0] - num_test - self.seq_len]
        border2s = [num_train, num_train + num_vali, data.shape[0]]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]
        self.slice = slice(border1, border2)
        
        df_stamp = generate_datetime_series(self.start_time_str, self.freq, data.shape[0], 'date')
        df_stamp = df_stamp[['date']][border1:border2]

        t_of_d, d_of_w = None, None
        time_ind = (df_stamp['date'].values - df_stamp['date'].values.astype('datetime64[D]')) / pd.to_timedelta(self.freq)
        t_of_d = time_ind.reshape([time_ind.shape[0],1])
        print('t_of_d.shape',t_of_d.shape)
        
        dow = df_stamp.date.apply(lambda row:row.dayofweek,1)
        d_of_w = dow.values.reshape([dow.shape[0],1])
        print('d_of_w.shape', d_of_w.shape)

        data_stamp = np.concatenate([t_of_d, d_of_w],axis=-1)
        data_stamp = data_stamp.astype(np.int64)
        print('data_stamp.shape:',data_stamp.shape)

        self.scaler.fit(data[border1s[0]:border2s[0]])
        data = self.scaler.transform(data)
        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]       
        self.data_stamp = data_stamp
        print(f'data.shape:',self.data_x.shape, self.data_y.shape)
        print(f'data_stamp.shape:',self.data_stamp.shape)

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        # seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_x_mark, seq_y

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)

