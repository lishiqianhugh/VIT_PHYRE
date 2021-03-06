# VIT PHYRE
This is an implementation of Vision Transformer on [PHYRE](https://phyre.ai/).
## How to use
* Environment
  * The project is developed and tested with python 3.6, pytorch 1.1 and cuda 11.6, but any version newer than that should work.
  * To get the pretrained VIT, you may want to install `timm` from Tsinghua open source using
  ```
  pip install timm -i https://pypi.tuna.tsinghua.edu.cn/simple
  ```
* Prepare dataset
  * Run dataset/phyreo.py to get train dataloader for faster training
  ```
  python dataset/phyreo.py
  ```
  * Before this, you may want to customize your train dataloader by modify `utils/config.py `


* Train VIT
  * There are two types of VIT to train (Pretrained / From scratch)
  * Run train.py to trian the protocal and fold you want, such as
  ```
  python train.py --protocal=within --fold=0 --batch_size=128
  ```
  
  * The log and model will be saved in `exp_{model_mode}_{data_mode}_{epoch}_{batch_size}_{lr}/`
* Evaluate
  * First you have to move the trained model to `models/`
  * Then run evaluate.py to test the model with specified protocal and fold
  ```
  python evaluate.py --model='Pretrained_VIT_PHYRE.pt' --protocal=within --fold=0 --batch_size=128*10
  ```
## Exp Record
Note that all experiments are based on one ball tier in [PHYRE](https://phyre.ai/).
### Within

| fold | Random | DQN    | RPIN  | epoch1 | epoch2 | epoch3 |
|------|--------|-------|-------|--------|--------|--------|  
| 0    | 13.44  | 76.82 | **85.49** | 75.63  | 80.20  | 79.62  |
| 1    | 14.01  | 79.72 | 86.57 | 78.60  | 81.67  | 83.45  |
| 2    | 13.79  | 78.22 | 85.58 |        |        |        |
| 3    | 13.80  | 75.86 | 84.11 |        |        |        |
| 4    | 12.75  | 77.03  | 85.30 |        |        |        |
| 5    | 13.34  | 78.42 | 85.18 |        |        |        |
| 6    | 13.95  | 78.01 | 84.78 |        |        |        |
| 7    | 14.30  | 77.34 | 84.32 |        |        |        |
| 8    | 13.36  | 78.04 | 85.71 |        |        |        |
| 9    | 14.33  | 76.87 | 85.17 |        |        |        |

### Cross

| fold    | Random | DQN   | RPIN  | epoch1    | epoch2 | epoch3    | epoch4 | epoch6 
|---------|--------|-------|-------|-----------|--------|-----------|--------|--------|  
| 0       | 11.78  | 43.69 | 50.86 | 46.99     | 48.21  | **52.50** |        | 43.43  |
| 1       | 12.42  | 30.96 | 36.58 | 27.08     |        | 31.89     | 33.53  |
| 2       | 18.18  | 43.05 | 55.44 |           |        | 53.93     | 50.83  |
| 3       | 12.42  | 43.91 | 38.34 | 34.36     | 29.11  | 34.42     | 34.10  |
| 4       | 3.81   | 22.77 | 37.11 | 34.13     | ...    |           |
| 5       | 22.50  | 44.40 | 47.23 | **51.11** |        | ...       |
| 6       | 11.73  | 34.53 | 38.23 | **38.36** | 32.87  |           |
| 7       | 13.29  | 39.20 | 47.19 | 45.13     |        |           |
| 8       | 8.94   | 18.98 | 32.23 | 26.70     | 27.42  | 26.46     |        | 26.18  |
| 9       | 14.60  | 46.46 | 38.76 |           | 35.17  |           | 
| Average | 13.0   | 36.8  | 42.2  | 40.53     |
