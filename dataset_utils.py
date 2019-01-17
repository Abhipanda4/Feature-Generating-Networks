from torch.utils.data.dataset import Dataset
from torchvision import transforms

import numpy as np
import os
import scipy.io as scio
from PIL import Image

class SUNDataset(Dataset):
    def __init__(self, root_dir="data", train=True, augment=False, model=None):
        super(SUNDataset, self).__init__()
        self.root_dir = root_dir
        self.image_dir = os.path.join(self.root_dir, "images")

        self.labels, self.train_dataset, self.test_dataset = self.create_orig_dataset()
        if train:
            self.dataset = self.train_dataset
        else:
            self.dataset = self.test_dataset

        self.augment = augment
        if self.augment:
            assert model is not None
            self.aug_dataset = self.create_aug_dataset(model)

        self.transformations = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
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
                train_dataset.extend(label_set[label]['images'])
            else:
                test_dataset.extend(label_set[label]['images'])

        return labels, train_dataset, test_dataset

    def create_aug_dataset(self, generator):
        pass

    def __getitem__(self, index):
        img_path = self.dataset[index]

        # load image
        img = Image.open(os.path.join(self.image_dir, img_path))
        if self.transformations:
            img = self.transformations(img)

        label = self.get_label_from_image(img_path)
        label_attr = self.labels[label]['attribute']
        label_idx = self.labels[label]['index']

        return img, label_attr, label_idx

    def __len__(self):
        return len(self.dataset)
