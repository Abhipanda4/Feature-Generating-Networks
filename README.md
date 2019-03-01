## Feature Generating Networks for Zero Shot Learning

PyTorch implementation of paper: <https://arxiv.org/abs/1712.00981>

Currently, only Animals With Attributes 2 dataset is supported(<https://cvml.ist.ac.at/AwA2/>)

<!--Accuracy obtained: 96.1%-->
Remarks:
* For training the model, use ``python3 main.py --n_epochs 20 --use_cls_loss --visualize``
* Using MLPClassifier instead of a linear Softmax classifier yields much better results.

Note: 
The dataset has to be downloaded and extracted into proper numpy arrays of specified shapes for the training/testing the model. See comments in datautils.py for more info.
