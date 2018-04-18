#### Introduction
This is the repo for our AAAI-18 paper "DarkRank: Accelerating Deep Metric Learning via Cross Sample Similarities" 
If you found this useful, please cite our paper.

#### Preparing MXNet
1. make a working directory
```
mkdir darkrank
```

2. clone the lastest stable mxnet
```
git clone --recursive https://github.com/apache/incubator-mxnet.git mxnet-darkrank
```

3. clone LSoftmax from luoyetx
```
git clone https://github.com/luoyetx/mx-lsoftmax.git lsoftmax
```

4. copy LSoftmax to mxnet
```
cp lsoftmax/operator/lsoftmax-inl.h lsoftmax/operator/lsoftmax.cc lsoftmax/operator/lsoftmax.cu mxnet-darkrank/src/operator/
```

5. install dependency and make mxnet
```
sudo apt-get update
sudo apt-get install -y build-essential git
sudo apt-get install -y libopenblas-dev liblapack-dev
sudo apt-get install -y libopencv-dev
```

```
cd mxnet-darkrank
make -j $(nproc) USE_OPENCV=1 USE_BLAS=openblas USE_CUDA=1 USE_CUDA_PATH=/usr/local/cuda USE_CUDNN=1
```

6. install mxnet
```
pip install -e mxnet-darkrank/python 
```

7. clone darkrank


#### Preparing Data
1. Download Market-1501-v15.09.15.zip from http://www.liangzheng.org/Project/project_reid.html
2. unzip Market-1501-v15.09.15.zip into `data` directory
3. run generate_rec.sh

#### Training
1. download imagenet pretrain
```
wget http://data.dmlc.ml/models/imagenet/inception-bn/Inception-BN-0126.params -O models/inception-bn-0126.params
wget http://data.dmlc.ml/models/imagenet/nin/nin-0000.params -O models/nin-0000.params
```
2. teacher baseline
```
python reid.py --mode inception-cls-LMNN-Market1501 --even-iter --data-dir data/Market-list --num-examples 13314 --num-id 751 --lr 0.01 --num-epochs 100 --train-file train --student-network inception-bn --student-params-prefix inception-bn --student-params-epoch 126 --gpus 0
```

3. student baseline
```
python reid.py --mode nin-cls-LMNN-Market1501 --even-iter --data-dir data/Market-list --num-examples 13314 --num-id 751 --lr 0.01 --num-epochs 100 --train-file train --student-network nin-head-bn --student-params-prefix nin --student-params-epoch 0 --gpus 0
```

4. darkrank
```
python reid.py --mode distill-listnet-Market1501 --even-iter --data-dir data/Market-list --num-examples 13314 --num-id 751 --num-epochs 100 --train-file train --student-network nin-head-bn --student-params-prefix nin --student-params-epoch 0 --teacher-network inception-bn --teacher-params-prefix inception-cls-LMNN-Market1501-1524036667 --teacher-params-epoch 100 --lr 5e-4 --score-power 2.0 --embedding-l2-norm 3.5 --list-length 4 --loss-weight-listnet 16.0 --gpus 0
```

#### Testing
extract 1024d `l2_norm_output` and do standard test
