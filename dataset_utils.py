import torch
from torch.utils.data.dataset import Dataset
from torchvision import transforms
from torch.autograd import Variable

import numpy as np
import os
import scipy.io as scio
from PIL import Image

class SUNDataset(Dataset):
    def __init__(self, device, root_dir="data", train=True, synthetic=False, syn_dataset=None):
        super(SUNDataset, self).__init__()
        self.device = device

        self.root_dir = root_dir
        self.image_dir = os.path.join(self.root_dir, "images")

        self.synthetic = synthetic
        if self.synthetic:
            assert syn_dataset is not None
            self.syn_dataset = syn_dataset
        else:
            self.labels, self.train_dataset, self.test_dataset = self.create_orig_dataset()
            if train:
                self.dataset = self.train_dataset
            else:
                self.dataset = self.test_dataset

        self.transformations = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

    def load_data(self):
        '''
        Returns:
            `label_set`: A dict whose key identifies the label for any image.
                Each label consists of 2 fields:
                * attributes: collection of per image attributes for
                              all images belonging to this label
                * images: paths for all images of this label
        '''

        assert os.path.exists(self.root_dir)
        image_mat = os.path.join(self.root_dir, "SUNAttributeDB/images.mat")
        image_arr = scio.loadmat(image_mat)["images"]

        attr_mat = os.path.join(self.root_dir, "SUNAttributeDB/attributeLabels_continuous.mat")
        attr_arr = scio.loadmat(attr_mat)["labels_cv"]

        label_set = dict()
        for img_info, attr in zip(image_arr, attr_arr):
            # find name of image
            img = img_info[0][0]
            label = self.get_label_from_image(img)

            if label not in label_set.keys():
                label_set[label] = dict()
                label_set[label]['attributes'] = []
                label_set[label]['images'] = []

            label_set[label]['attributes'].append(attr)
            label_set[label]['images'].append(img)

        return label_set

    def get_label_from_image(self, img_path):
        label = "/".join(img_path.split('/')[:-1])
        return label

    def create_orig_dataset(self):
        train_dataset = []
        test_dataset = []
        labels = dict()
        label_set = self.load_data()

        keys = list(label_set.keys())
        for idx, label in enumerate(keys):
            labels[label] = dict()
            labels[label]['attribute'] = \
                    np.mean(np.asarray(label_set[label]['attributes']), axis=0)
            labels[label]['index'] = idx

            # use standard split
            if idx < 645:
                # 15 images for train set and 5 images for test - GZSL
                train_dataset.extend(label_set[label]['images'][:15])
                test_dataset.extend(label_set[label]['images'][15:])
            else:
                test_dataset.extend(label_set[label]['images'])

        return labels, train_dataset, test_dataset

    def __getitem__(self, index):
        if self.synthetic:
            # choose an example from syn_dataset
            img_features, label_attr, label_idx = self.syn_dataset[index]
            return img_features, label_attr, label_idx
        else:
            img_path = self.dataset[index]
            # load image from path
            img = Image.open(os.path.join(self.image_dir, img_path)).convert("RGB")
            if self.transformations:
                img = self.transformations(img)

            label = self.get_label_from_image(img_path)
            label_attr = self.labels[label]['attribute']
            label_idx = self.labels[label]['index']

            return img, label_attr, label_idx

    def __len__(self):
        if self.synthetic:
            return len(self.syn_dataset)
        else:
            return len(self.dataset)
