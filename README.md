## Feature Generating Networks for Zero Shot Learning

PyTorch implementation of paper: <https://arxiv.org/abs/1712.00981>

4 datasets are currently supported: SUN, CUB, AWA1 & AWA2.

<!--Accuracy obtained: 96.1%-->
Remarks:
* For training the model, use ``python3 main.py --n_epochs 20 --use_cls_loss``
* Using MLPClassifier instead of a linear Softmax classifier yields much better results.

Note: 
The dataset has to be downloaded and extracted into proper numpy arrays of specified shapes for training/testing the model. All relevant files except the Resnet101 feature matrix have been uploaded in this repo. See comments in datautils.py for more info.