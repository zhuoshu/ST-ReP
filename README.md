This the official implementation of ST-ReP: Learning Predictive Representations Efficiently for Spatial-Temporal Forecasting (AAAI 2025). [[Paper](https://ojs.aaai.org/index.php/AAAI/article/view/33465)]

## Requirements

The code is built based on Python 3.9, PyTorch 1.13.1.
You can install PyTorch following the instruction in [PyTorch](https://pytorch.org/get-started/locally/). For example:

```bash
pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117
```

After ensuring that PyTorch is installed correctly, you can install other dependencies via:

```bash
pip install -r requirements.txt
```

## Data Preparation
Following the detail descriptions [here](datasets/readme.md).

## Pre-Training and Downstream forecasting
The scripts for reproduction of ST-ReP and two simple baselines HL and Ridge are presented as follows.

### PEMS04
```python
python run.py --config_file PEMS04.yaml --modelid STReP --device cuda:0
python run.py --config_file PEMS04.yaml --modelid HL
python run.py --config_file PEMS04.yaml --modelid Ridge 
```
### PEMS08
```python
python run.py --config_file PEMS08.yaml --modelid STReP --device cuda:0
python run.py --config_file PEMS08.yaml --modelid HL
python run.py --config_file PEMS08.yaml --modelid Ridge 
```
### Temperature
```python
python run.py --config_file temperature2016.yaml --modelid STReP --device cuda:0
python run.py --config_file temperature2016.yaml --modelid HL
python run.py --config_file temperature2016.yaml --modelid Ridge 
```
### Humidity
```python
python run.py --config_file humidity2016.yaml --modelid STReP --device cuda:0
python run.py --config_file humidity2016.yaml --modelid HL
python run.py --config_file humidity2016.yaml --modelid Ridge 
```
### SDWPF
```python
python run.py --config_file SDWPF.yaml --modelid STReP --device cuda:0
python run.py --config_file SDWPF.yaml --modelid HL
python run.py --config_file SDWPF.yaml --modelid Ridge 
```
### CA
```python
python run.py --config_file CA.yaml --modelid STReP --device cuda:0
python run.py --config_file CA.yaml --modelid HL
python run.py --config_file CA.yaml --modelid Ridge 
```

Results will be saved in the `EXP_results` folder.

## Acknowledgement
Thanks to the following inspiring research and their valuable codes.
* TS2Vec (https://github.com/zhihanyue/ts2vec)
* Autoformer (https://github.com/thuml/Autoformer)
* STEP (https://github.com/GestaltCogTeam/STEP)

## Cite
If you find this project helpful, please cite us:
```bibtex
@article{STReP_2025,
title={ST-ReP: Learning Predictive Representations Efficiently for Spatial-Temporal Forecasting},
volume={39},
url={https://ojs.aaai.org/index.php/AAAI/article/view/33465},
DOI={10.1609/aaai.v39i12.33465},
number={12},
journal={Proceedings of the AAAI Conference on Artificial Intelligence},
author={Zheng, Qi and Yao, Zihao and Zhang, Yaying},
year={2025},
month={Apr.},
pages={13419-13427}
}
```
