import torch
from torch.utils.data import DataLoader
import argparse

from datautils import AwA2Dataset
from trainer import Trainer

parser = argparse.ArgumentParser()

parser.add_argument('--dataset', type=str, default='awa2')
parser.add_argument('--latent_dim', type=int, default=128)
parser.add_argument('--n_critic', type=int, default=5)
parser.add_argument('--lmbda', type=float, default=10.0)
parser.add_argument('--beta', type=float, default=0.01)
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--n_epochs', type=int, default=10)
parser.add_argument('--use_cls_loss', action='store_true', default=False)

args = parser.parse_args()

if args.dataset == 'awa2':
    x_dim = 2048
    attr_dim = 85
    train_classes = 40
    test_classes = 10
else:
    raise NotImplementedError

n_epochs = args.n_epochs

# trainer object for mini batch training
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
train_agent = Trainer(
    device, x_dim, args.latent_dim, attr_dim,
    train_classes, test_classes,
    args.n_critic, args.lmbda, args.beta,
    args.batch_size
)

params = {
    'batch_size': args.batch_size,
    'shuffle': True,
    'num_workers': 0,
    'drop_last': True
}

train_dataset = AwA2Dataset(device, train_classes, test_classes)
train_generator = DataLoader(train_dataset, **params)

# =============================================================
# PRETRAIN THE SOFTMAX CLASSIFIER
# =============================================================
model_name = "disc_classifier"
success = train_agent.load_model(model=model_name)
if success:
    print("Discriminative classifier parameters loaded...")
else:
    print("Training the discriminative classifier...")
    for ep in range(1, n_epochs + 1):
        loss = 0
        for idx, (img_features, label_attr, label_idx) in enumerate(train_generator):
            l = train_agent.fit_classifier(img_features, label_attr, label_idx)
            loss += l

        print("Loss for epoch: %3d - %.4f" %(ep, loss))

    train_agent.save_model(model=model_name)

# =============================================================
# TRAIN THE GANs
# =============================================================
model_name = "gan"
success = train_agent.load_model(model=model_name)
if success:
    print("\nGAN parameters loaded....")
else:
    print("\nTraining the GANS...")
    for ep in range(1, n_epochs + 1):
        loss_dis = 0
        loss_gan = 0
        for idx, (img_features, label_attr, label_idx) in enumerate(train_generator):
            l_d, l_g = train_agent.fit_GAN(img_features, label_attr, label_idx, args.use_cls_loss)
            loss_dis += l_d
            loss_gan += l_g

        print("Loss for epoch: %3d - D: %.4f | G: %.4f"\
                %(ep, loss_dis, loss_gan))

    train_agent.save_model(model=model_name)

# =============================================================
# TRAIN FINAL CLASSIFIER ON SYNTHETIC + REAL DATASET
# =============================================================

# create new synthetic dataset using trained Generator
syn_dataset = train_agent.create_syn_dataset(train_dataset.test_labels, train_dataset.attributes)
synthetic_train_dataset = AwA2Dataset(device, train_classes, test_classes, synthetic=True, syn_dataset=syn_dataset)
syn_train_generator = DataLoader(synthetic_train_dataset, **params)

model_name = "final_classifier"
success = train_agent.load_model(model=model_name)
if success:
    print("\nFinal classifier parameters loaded....")
else:
    print("\nTraining the final classifier on the synthetic dataset...")
    for ep in range(1, n_epochs + 1):
        syn_loss = 0
        for idx, (img, label_attr, label_idx) in enumerate(syn_train_generator):
            l = train_agent.fit_final_classifier(img, label_attr, label_idx)
            syn_loss += l

        # print losses on real and synthetic datasets
        print("Loss for epoch: %3d - %.4f" %(ep, syn_loss))

    train_agent.save_model(model=model_name)


# Testing phase
test_dataset = AwA2Dataset(device, train_classes, test_classes, train=False)
test_generator = DataLoader(test_dataset, **params)
print("\nZSL Accuracy: %.3f" % train_agent.test(test_generator))
