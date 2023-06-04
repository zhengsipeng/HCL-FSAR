# HCL
This is the official implementation for the paper "Few-shot Action Recognition with Hierarchical Contrastive Learning"

## Installation
**1. Clone and enter this repository**
**2. Install packages for Python 3.7**
```
pip -r requirements.txt
```
**3. Install apex**
```
git clone https://github.com/NVIDIA/apex
cd apex
pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
cd ..
```


## Data Preparation

**1. Dataset**:
Our work carries out experiments on four datasets. Download and unpack these datasets in the "data/videos" directory ([Kinetics](https://deepmind.com/research/open-source/kinetics), [SomethingV2](https://paperswithcode.com/dataset/something-something-v2), [HMDB51](https://serre-lab.clps.brown.edu/resource/hmdb-a-large-human-motion-database/) and [UCF101](https://www.crcv.ucf.edu/research/data-sets/ucf101/)). The dataset dir should be like:
```
|-- data
    |-- videos
    |   |-- ssv2_100_otam
    |   |   | -- train
    |   |   |   | -- ***.webm
    |   |   |   | -- ...
    |   |   | -- val
    |   |   | -- test
    |   |-- ssv2_100
    |   |   | -- train
    |   |   | -- val
    |   |   | -- test
    |   |-- kinetics100
    |   |   | -- train
    |   |   | -- val
    |   |   | -- test
    |   |-- hmdb51
    |   |   | -- class0
    |   |   |    | -- ***.avi
    |   |   | -- class1
    |   |   | -- ...
    |   |-- ucf101
    |   |   | -- class0
    |   |   |    | -- ***.avi
    |   |   | -- class1
    |   |   | -- ...
```

**2. Data Splits**:
We use 5 data splits, which have already been provided in "data/splits".
```
train/val/test.list are split video list
train/val/test_class.txt are split class list
``` 

**3. Extract Frames**:
Extract frames for videos in the datasets by running. For example, run the following command for SSv2-OTAM:
```
python extract_frames.py ssv2_100_otam
```
The final image dirs should be like:
```
|-- data
    |-- images
    |   |-- ssv2_100_otam
    |   |   |-- train0
    |   |   |   |-- 197
    |   |   |   |   |-- 1.jpg
    |   |   |   |   |-- ...
    |   |   |   |-- ...
    |   |-- ssv2_100
    |   |-- kinetics100
    |   |-- hmdb51
    |   |-- ucf101
```

## Models and Embeddings
**1. ResNet50 pre-trained parameters is available at**
```
https://www.dropbox.com/s/95aqr6zarj4fuv7/resnet50-19c8e357.pth?dl=0
```

```
cp models/resnet50-19c8e357.pth  /root/.cache/torch/hub/checkpoints/
```

**2. Checkpints**

We provide pre-trained checkpoints on [SSv2_100_OTAM](https://www.dropbox.com/s/tg96kmunig5fdjw/hcl_ssv2_100_otam.pth?dl=0), then move them in ``models'' dir.
```
cp ***.pth models/
```

The results of our provided checkpoints are comparable with  what we report in the paper.

| Dataset  | 1-shot  | 2-shot | 3-shot | 4-shot | 5-shot |
|  ----  | ----  |  ----  | ----  | ----  | ----  |
| SSv2_100_OTAM  | 47.73 | 55.41 | 60.02 | 62.71 | 64.89 |

**3. BERT embeddings**
We provide BERT embeddings of [SSv2_100](https://www.dropbox.com/s/5ynq5458o3pclnv/ssv2_100_embs.pkl?dl=0); [SSv2_100_OTAM](https://www.dropbox.com/s/tg96kmunig5fdjw/hcl_ssv2_100_otam.pth?dl=0) and Kinetics,HMDB51,UCF101 in 
```
bert_model/***_embs.pkl
```

## Evaluation of our model
For example, to evaluate our model on 1-shot setup of SSv2_100_Otam, directly run the following command:
```
python eval.py --ckpt_path models/hcl_ssv2_100_otam.pth 
    --dataset ssv2_100_otam 
    --method hcl 
    --use_spatial 
    --sigma_global 1 
    --sigma_temp 0.5 
    --sigma_spa 0.5 
    --data_aug type1 
    --enc_layers 1 
    --d_model 1152 
    --num_gpus 4
    --shot 1
```

## Training HCL
For example, to train our model on ssv2_100_otam, using:
```
sh scripts/train_global_temp_spa.sh
```