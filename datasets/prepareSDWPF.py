import pandas as pd
import numpy as np

# Handle outliers according to Section 4.1 in the original paper: 
#  [SDWPF: A Dataset for Spatial Dynamic Wind Power Forecasting Challenge at KDD Cup 2022] (http://arxiv.org/abs/2208.04360)
# Note that we only use the values of column 'Patv' to build datasets.
def handle_exceptions(df): 
    mask = pd.Series(index=df.index, data=0) 
    mask |= (df['Wspd'] > 2.5) & (df['Patv'] <= 0)
    mask |= (df['Pab1'] > 89) | (df['Pab2'] > 89) | (df['Pab3'] > 89)    
    mask |= df['Patv'].isna()
    mask |= (df['Patv'] < 0)
    mask |= (df['Ndir'] > 720) | (df['Ndir'] < -720)
    mask |= (df['Wdir'] > 180) | (df['Wdir'] < -180)
    df['Patv_clear'] = df['Patv'].where(~mask, 0)
    return mask

df = pd.read_csv('./SDWPF/raw_data/wtbdata_245days.csv')
pa_mask = handle_exceptions(df)


df_ready = df[['TurbID','Patv_clear']]
df_list = df_ready.groupby(['TurbID'])
data = []
for k,v in df_list:
    data.append(v['Patv_clear'].values)

data = np.stack(data, 1)
data = np.expand_dims(data,axis=-1)
print(data.shape)
np.savez_compressed('./SDWPF/SDWPF.npz',data=data)
print('npz file saved')