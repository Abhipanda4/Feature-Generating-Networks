import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import uniform, normal

import os
import numpy as np
from sklearn.metrics import accuracy_score

from models import Generator, Discriminator, MLPClassifier, Resnet101

class Trainer:
    def __init__(
        self, device, x_dim, z_dim, attr_dim, **kwargs):
        '''
        Trainer class.
        Args:
            device: CPU/GPU
            x_dim: Dimension of image feature vector
            z_dim: Dimension of noise vector
            attr_dim: Dimension of attribute vector
            kwargs
        '''
        self.device = device

        self.x_dim = x_dim
        self.z_dim = z_dim
        self.attr_dim = attr_dim

        self.n_critic = kwargs.get('n_critic', 5)
        self.lmbda = kwargs.get('lmbda', 10.0)
        self.beta = kwargs.get('beta', 0.01)
        self.bs = kwargs.get('batch_size', 32)

        self.gzsl = kwargs.get('gzsl', False)
        self.n_train = kwargs.get('n_train')
        self.n_test = kwargs.get('n_test')
        if self.gzsl:
            self.n_test = self.n_train + self.n_test

        self.eps_dist = uniform.Uniform(0, 1)
        self.Z_dist = normal.Normal(0, 1)

        self.eps_shape = torch.Size([self.bs, 1])
        self.z_shape = torch.Size([self.bs, self.z_dim])

        self.net_G = Generator(self.z_dim, self.attr_dim).to(self.device)
        self.optim_G = optim.Adam(self.net_G.parameters(), lr=1e-4)

        self.net_D = Discriminator(self.x_dim, self.attr_dim).to(self.device)
        self.optim_D = optim.Adam(self.net_D.parameters(), lr=1e-4)

        # classifier for judging the output of generator
        self.classifier = MLPClassifier(
            self.x_dim, self.attr_dim, self.n_train
        ).to(self.device)
        self.optim_cls = optim.Adam(self.classifier.parameters(), lr=1e-4)

        # Final classifier trained on augmented data for GZSL
        self.final_classifier = MLPClassifier(
            self.x_dim, self.attr_dim, self.n_test
        ).to(self.device)
        self.optim_final_cls = optim.Adam(self.final_classifier.parameters(), lr=1e-4)

        self.criterion_cls = nn.CrossEntropyLoss()

        self.model_save_dir = "saved_models"
        if not os.path.exists(self.model_save_dir):
            os.mkdir(self.model_save_dir)

    def get_conditional_input(self, X, C_Y):
        new_X = torch.cat([X, C_Y], dim=1).float()
        return autograd.Variable(new_X).to(self.device)

    def fit_classifier(self, img_features, label_attr, label_idx):
        '''
        Train the classifier in supervised manner on a single
        minibatch of available data
        Args:
            img         : bs X 2048
            label_attr  : bs X 85
            label_idx   : bs
        Returns:
            loss for the minibatch
        '''
        img_features = autograd.Variable(img_features).to(self.device)
        label_attr = autograd.Variable(label_attr).to(self.device)
        label_idx = autograd.Variable(label_idx).to(self.device)

        X_inp = self.get_conditional_input(img_features, label_attr)
        Y_pred = self.classifier(X_inp)

        self.optim_cls.zero_grad()
        loss = self.criterion_cls(Y_pred, label_idx)
        loss.backward()
        self.optim_cls.step()

        return loss.item()

    def get_gradient_penalty(self, X_real, X_gen):
        eps = self.eps_dist.sample(self.eps_shape).to(self.device)
        X_penalty = eps * X_real + (1 - eps) * X_gen

        X_penalty = autograd.Variable(X_penalty, requires_grad=True).to(self.device)
        critic_pred = self.net_D(X_penalty)
        grad_outputs = torch.ones(critic_pred.size()).to(self.device)
        gradients = autograd.grad(
                outputs=critic_pred, inputs=X_penalty,
                grad_outputs=grad_outputs,
                create_graph=True, retain_graph=True, only_inputs=True
        )[0]

        grad_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        return grad_penalty

    def fit_GAN(self, img_features, label_attr, label_idx, use_cls_loss=True):
        L_gen = 0
        L_disc = 0
        total_L_disc = 0

        img_features = autograd.Variable(img_features.float()).to(self.device)
        label_attr = autograd.Variable(label_attr.float()).to(self.device)
        label_idx = label_idx.to(self.device)

        # =============================================================
        # optimize discriminator
        # =============================================================
        X_real = self.get_conditional_input(img_features, label_attr)
        for _ in range(self.n_critic):
            Z = self.Z_dist.sample(self.z_shape).to(self.device)
            Z = self.get_conditional_input(Z, label_attr)

            X_gen = self.net_G(Z)
            X_gen = self.get_conditional_input(X_gen, label_attr)

            # calculate normal GAN loss
            L_disc = (self.net_D(X_gen) - self.net_D(X_real)).mean()

            # calculate gradient penalty
            grad_penalty = self.get_gradient_penalty(X_real, X_gen)
            L_disc += self.lmbda * grad_penalty

            # update critic params
            self.optim_D.zero_grad()
            L_disc.backward()
            self.optim_D.step()

            total_L_disc += L_disc.item()

        # =============================================================
        # optimize generator
        # =============================================================
        Z = self.Z_dist.sample(self.z_shape).to(self.device)
        Z = self.get_conditional_input(Z, label_attr)

        X_gen = self.net_G(Z)
        X = torch.cat([X_gen, label_attr], dim=1).float()
        L_gen = -1 * torch.mean(self.net_D(X))

        if use_cls_loss:
            self.classifier.eval()
            Y_pred = F.softmax(self.classifier(X), dim=0)
            log_prob = torch.log(torch.gather(Y_pred, 1, label_idx.unsqueeze(1)))
            L_cls = -1 * torch.mean(log_prob)
            L_gen += self.beta * L_cls

        self.optim_G.zero_grad()
        L_gen.backward()
        self.optim_G.step()

        return total_L_disc, L_gen.item()

    def fit_final_classifier(self, img_features, label_attr, label_idx):
        img_features = autograd.Variable(img_features.float()).to(self.device)
        label_attr = autograd.Variable(label_attr.float()).to(self.device)
        label_idx = label_idx.to(self.device)

        X_inp = self.get_conditional_input(img_features, label_attr)
        Y_pred = self.final_classifier(X_inp)

        self.optim_final_cls.zero_grad()
        loss = self.criterion_cls(Y_pred, label_idx)
        loss.backward()
        self.optim_final_cls.step()

        return loss.item()

    def create_syn_dataset(self, test_labels, attributes, seen_dataset, n_examples=400):
        '''
        Creates a synthetic dataset based on attribute vectors of unseen class
        Args:
            test_labels: A dict with key as original serial number in provided
                dataset and value as the index which is predicted during
                classification by network
            attributes: A np array containing class attributes for each class
                of dataset
            seen_dataset: A list of 3-tuple (x, orig_label, y) where x belongs to one of the
                seen classes and y is classification label. Used for generating
                latent representations of seen classes in GZSL
            n_samples: Number of samples of each unseen class to be generated(Default: 400)
        Returns:
            A list of 3-tuple (z, _, y) where z is latent representations and y is
        '''
        syn_dataset = []
        for test_cls, idx in test_labels.items():
            attr = attributes[test_cls - 1]
            z = self.Z_dist.sample(torch.Size([n_examples, self.z_dim]))
            c_y = torch.stack([torch.FloatTensor(attr) for _ in range(n_examples)])

            z_inp = self.get_conditional_input(z, c_y)
            X_gen = self.net_G(z_inp)

            syn_dataset.extend([(X_gen[i], test_cls, idx) for i in range(n_examples)])

        if seen_dataset is not None:
            syn_dataset.extend(seen_dataset)

        return syn_dataset

    def test(self, data_generator, pretrained=False):
        if pretrained:
            model = self.classifier
        else:
            model = self.final_classifier

        # eval mode
        model.eval()
        batch_accuracies = []
        for idx, (img_features, label_attr, label_idx) in enumerate(data_generator):
            img_features = img_features.to(self.device)
            label_attr = label_attr.to(self.device)

            X_inp = self.get_conditional_input(img_features, label_attr)
            with torch.no_grad():
                Y_probs = model(X_inp)
            _, Y_pred = torch.max(Y_probs, dim=1)

            Y_pred = Y_pred.cpu().numpy()
            Y_real = label_idx.cpu().numpy()

            acc = accuracy_score(Y_pred, Y_real)
            batch_accuracies.append(acc)
        return np.mean(batch_accuracies)

    def save_model(self, model=None):
        if "disc_classifier" in model:
            ckpt_path = os.path.join(self.model_save_dir, model + ".pth")
            torch.save(self.classifier.state_dict(), ckpt_path)

        elif "gan" in model:
            dset_name = model.split('_')[0]
            g_ckpt_path = os.path.join(self.model_save_dir, "%s_generator.pth" % dset_name)
            torch.save(self.net_G.state_dict(), g_ckpt_path)

            d_ckpt_path = os.path.join(self.model_save_dir, "%s_discriminator.pth" % dset_name)
            torch.save(self.net_D.state_dict(), d_ckpt_path)

        elif "final_classifier" in model:
            ckpt_path = os.path.join(self.model_save_dir, model + ".pth")
            torch.save(self.final_classifier.state_dict(), ckpt_path)

        else:
            raise Exception("Trying to save unknown model: %s" % model)

    def load_model(self, model=None):
        if "disc_classifier" in model:
            ckpt_path = os.path.join(self.model_save_dir, model + ".pth")
            if os.path.exists(ckpt_path):
                self.classifier.load_state_dict(torch.load(ckpt_path))
                return True

        elif "gan" in model:
            f1, f2 = False, False
            dset_name = model.split('_')[0]
            g_ckpt_path = os.path.join(self.model_save_dir, "%s_generator.pth" % dset_name)
            if os.path.exists(g_ckpt_path):
                self.net_G.load_state_dict(torch.load(g_ckpt_path))
                f1 = True

            d_ckpt_path = os.path.join(self.model_save_dir, "%s_discriminator.pth" % dset_name)
            if os.path.exists(d_ckpt_path):
                self.net_D.load_state_dict(torch.load(d_ckpt_path))
                f2 = True

            return f1 and f2

        elif "final_classifier" in model:
            ckpt_path = os.path.join(self.model_save_dir, model + ".pth")
            if os.path.exists(ckpt_path):
                self.final_classifier.load_state_dict(torch.load(ckpt_path))
                return True

        else:
            raise Exception("Trying to load unknown model: %s" % model)

        return False
