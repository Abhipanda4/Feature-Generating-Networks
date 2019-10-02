## Feature Generating Networks for ZSL in Pytorch

PyTorch implementation of paper: [Feature Generating Networks for Zero-Shot Learning](https://arxiv.org/abs/1712.00981)

4 datasets are currently supported: SUN, CUB, AWA1 & AWA2. All datasets can be downloaded [here](http://datasets.d2.mpi-inf.mpg.de/xian/xlsa17.zip).

#### IMPORTANT:
The downloaded zip will have many files for each dataset, but we only require 2 files ``res101.mat`` & ``att_splits.mat``. Move these 2 files per dataset to the appropriate folder in this repo before starting to train/test.

* For training the model, use:
 ```python3 main.py --n_epochs 20 --use_cls_loss```

 All trainable parameters are saved in a folder named ``saved_models`` at the end of every epoch.