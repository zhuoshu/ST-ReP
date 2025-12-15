import xarray as xr
import matplotlib.pyplot as plt
import numpy as np

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--dataset_name", type=str, default='temperature2016')
args = parser.parse_args()

raw_data_path = {'temperature2016': './temperature2016/raw_data/*.nc',
                 'humidity2016': './humidity2016/raw_data/*.nc'}[args.dataset_name]
attr = {'temperature2016': 't2m', 
        'humidity2016': 'r'}[args.dataset_name]

rdata = xr.open_mfdataset(raw_data_path, combine='by_coords')


raw_data = rdata.__getattr__(attr)
print(raw_data.shape)

if len(raw_data.shape) == 4: # when there are different level, we choose the 13-th level which is sea level, following [AAAI 2022 CLCRN] (https://github.com/EDAPINENUT/CLCRN/tree/main)
    raw_data = raw_data[:,-1,...]

start_date = '2016-01-01'
end_date = '2017-01-01'
time = rdata.time.values
mask_dataset = np.bitwise_and(np.datetime64(
    start_date) <= time, time < np.datetime64(end_date))
data = raw_data[mask_dataset]
time_data = rdata.time[mask_dataset]
print(data.shape)

data = np.array(data)
data = data.reshape((data.shape[0], -1))
data = np.expand_dims(data, axis=-1)
print(data.shape)
np.savez_compressed(f'./{args.dataset_name}/{args.dataset_name}.npz',data=data)
print('npz file saved')
