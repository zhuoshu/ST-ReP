### PEMS04 and PEMS08
The PEMS datasets (PEMS04, and PEMS08) are available at [ASTGNN](https://github.com/guoshnBJTU/ASTGNN) or [STSGCN](https://github.com/Davidham3/STSGCN), and the data files (`PEMS04.npz` and `PEMS08.npz`) should be put into the `datasets/PEMS04` folder and the `datasets/PEMS08` folder.

### CA
The CA dataset can be obtained by referring to [LargeST](https://github.com/liuxu77/LargeST).  The data file `ca_his_2019.h5` should be put into the `datasets/ca` folder.

### Temperature and Humidity
The raw data of Temperature (`5.625deg/2m_temperature`) and Humidity (`5.625deg/relative_humidity`) can be downloaded from [WeatherBench](https://github.com/pangeo-data/WeatherBench)([Arxiv](https://arxiv.org/abs/2002.00469)). They should be put into the `datasets/temperature2016/raw_data` folder and the `datasets/humidity2016/raw_data` folder.
Then the following python scripts needs to be executed to obtain the npz file for training.
 ```python
    python prepareWeatherBench.py --dataset_name temperature2016
    python prepareWeatherBench.py --dataset_name humidity2016
 ```

### SDWPF
 The raw data of SDWPF `wtbdata_245days.csv` can be downloaded from [the Baidu KDD Cup 2022 website](https://aistudio.baidu.com/competition/detail/152/0/introduction) , and it should be put into the `datasets/SDWPF/raw_data` folder. Then the following python script needs to be executed to obtain the npz file for training.
 ```python
    python prepareSDWPF.py
 ```

