import torch
from torch.utils.data.dataset import Dataset
from torchvision import transforms
from torch.autograd import Variable

import numpy as np

class AwA2Dataset(Dataset):
    def __init__(self, device, n_train, n_test, train=True, synthetic=False, syn_dataset=None):
        '''
        Dataset for Animals with Attributes consists of 37322 images of 50 animals
        Args:
            device: torch.device object to use GPU/CPU
        '''
        super(AwA2Dataset, self).__init__()
        self.device = device
        self.n_train = n_train
        self.n_test = n_test

        # a np array of size (37322, 2048)
        self.features = np.load('./data/features.npy')
        # a np array of size (37322,)
        self.labels = np.load('./data/labels.npy')
        # a np array of size (50, 85)
        self.attributes = np.load('./data/attributes.npy')

        # file with all class names for deciding train/test split
        self.class_names = './data/classes.txt'

        self.synthetic = synthetic
        if self.synthetic:
            assert syn_dataset is not None
            self.syn_dataset = syn_dataset
        else:
            self.train_dataset, self.test_dataset = self.create_orig_dataset()
            if train:
                self.dataset = self.train_dataset
            else:
                self.dataset = self.test_dataset

    def get_label_maps(self):
        '''
        Returns the labels of all classes to be used as test set
        as described in proposed split
        '''
        test_classes = ['sheep','dolphin','bat','seal','blue+whale', 'rat','horse','walrus','giraffe','bobcat']
        with open(self.class_names) as fp:
            all_classes = fp.readlines()

        test_count = 0
        train_count = 0

        train_labels = dict()
        test_labels = dict()
        for line in all_classes:
            idx, name = [i.strip() for i in line.split(' ')]
            if name in test_classes:
                test_labels[int(idx)] = test_count
                test_count += 1
            else:
                train_labels[int(idx)] = train_count
                train_count += 1

        return train_labels, test_labels

    def create_orig_dataset(self):
        '''
        Partitions all 37322 image features into train/test based on proposed split
        Returns 2 lists, train_set & test_set: each entry of list is a 3-tuple
        (feature, label_in_dataset, label_for_classification)
        '''
        self.train_labels, self.test_labels = self.get_label_maps()
        train_set, test_set  = [], []

        for feat, label in zip(self.features, self.labels):
            if label in self.test_labels.keys():
                test_set.append((feat, label, self.test_labels[label]))
            else:
                train_set.append((feat, label, self.train_labels[label]))

        return train_set, test_set

    def __getitem__(self, index):
        if self.synthetic:
            # choose an example from synthetic dataset
            img_feature, orig_label, label_idx = self.syn_dataset[index]
        else:
            # choose an example from original dataset
            img_feature, orig_label, label_idx = self.dataset[index]

        label_attr = self.attributes[orig_label - 1]
        return img_feature, label_attr, label_idx

    def __len__(self):
        if self.synthetic:
            return len(self.syn_dataset)
        else:
            return len(self.dataset)
