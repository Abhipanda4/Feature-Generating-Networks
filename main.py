import torch
from torch.utils.data import DataLoader
import argparse

from dataset_utils import SUNDataset
from trainer import Trainer

parser = argparse.ArgumentParser()

parser.add_argument('--dataset', type=str, default='sun')
parser.add_argument('--latent_dim', type=int, default=128)
parser.add_argument('--n_critic', type=int, default=5)
parser.add_argument('--lmbda', type=float, default=10.0)
parser.add_argument('--beta', type=float, default=0.01)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--n_epochs', type=int, default=50)

args = parser.parse_args()

if args.dataset == 'sun':
    x_dim = 2048
    attr_dim = 102
    train_classes = 645
    test_classes = 72
else:
    raise NotImplementedError

params = {
    'batch_size': args.batch_size,
    'shuffle': True,
    'num_workers': 6
}
train_dataset = SUNDataset()
train_generator = DataLoader(train_dataset, **params)

train_agent = Trainer(
    args.x_dim, args.latent_dim, attr_dim,
    train_classes, train_classes + test_classes,
    args.n_critic, args.lmbda, args.beta,
    args.batch_size
)

n_epochs = args.n_epochs
n_epochs = 1
for ep in range(n_epochs):
    for idx, (img, label_attr, label_idx) in enumerate(train_generator):
        l = train_agent.fit_classifier(img, label_attr, label_idx)
        print(l)
        break

for ep in range(n_epochs):
    for idx, (img, label_attr, label_idx) in enumerate(train_generator):
        l_d, l_g = train_agent.fit_GAN(img, label_attr, label_idx)
        print(l_d, l_g)
        break

augmented_train_dataset = SUNDataset(augment=True, model=train_agent.net_G)
train_generator = DataLoader(augmented_train_dataset, **params)

# train a softmax classifier on the augmented dataset
for ep in range(n_epochs):
    for idx, (img, label_attr, label_idx) in enumerate(train_generator):
        l = train_agent.fit_final_classifier(img, label_attr, label_idx)
        print(l)
        break

test_dataset = SUNDataset(train=False)
