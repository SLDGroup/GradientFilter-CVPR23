import logging
import os
import numpy as np
import torch as th
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
from pytorch_lightning import LightningDataModule
from .sampling import mnist_iid, mnist_noniid, cifar_iid, cifar_noniid, tiny_imagenet_iid, tiny_imagenet_noniid, imagenet_iid, imagenet_noniid
from tqdm import tqdm


class DatasetSplit(Dataset):
    """An abstract Dataset class wrapped around Pytorch Dataset class.
    """

    def __init__(self, dataset, idxs, transform=None):
        self.dataset = dataset
        self.idxs = [int(i) for i in idxs]
        self.transform = transform

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        if self.transform is not None:
            image = self.transform(image)
        return {'image': image, 'label': label}


class ClsDataset(LightningDataModule):
    def __init__(self, data_dir, name='mnist',
                 num_partitions=2, iid=False, train_split=0.8,
                 batch_size=32, train_shuffle=True,
                 width=224, height=224,
                 train_workers=4, val_workers=1,
                 usr_group=None, partition=0, shards_per_partition=2):
        super(ClsDataset, self).__init__()
        self.name = name
        print(self.name)
        self.data_dir = data_dir
        self.num_partitions = num_partitions
        self.iid = iid
        self.current_partition = partition
        if usr_group is not None:
            grp = np.load(usr_group)
            self.usr_group = {v[0]: v[1:] for v in grp}
            assert len(self.usr_group) == self.num_partitions
        else:
            self.usr_group = None
        self.train_split = train_split
        self.train_dataset = None
        self.val_dataset = None
        self.train_workers = train_workers
        self.val_workers = val_workers
        self.batch_size = batch_size
        self.train_shuffle = train_shuffle
        self.width = width
        self.height = height
        self.shards_per_partition = shards_per_partition
        self.num_classes = 0

    def switch_partition(self, partition):
        logging.info(f"Load partition {partition}")
        update_usr_group = self.usr_group is None
        if update_usr_group:
            logging.warn("No partition def found")
        if self.name == 'mnist':
            apply_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Resize((self.width, self.height)),
                transforms.Normalize((0.1307,), (0.3081,))])
            raw_dataset = datasets.MNIST(self.data_dir, train=True, download=True,
                                         transform=apply_transform)
            self.num_classes = 10
            if self.usr_group is None:
                if self.iid:
                    self.usr_group = mnist_iid(
                        raw_dataset, self.num_partitions)
                else:
                    self.usr_group = mnist_noniid(
                        raw_dataset, self.num_partitions)
                for usr in self.usr_group.keys():
                    np.random.shuffle(self.usr_group[usr])
        elif self.name == 'fmnist':
            apply_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Resize((self.width, self.height)),
                transforms.Normalize((0.1307,), (0.3081,))])
            raw_dataset = datasets.FashionMNIST(self.data_dir, train=True, download=True,
                                                transform=apply_transform)
            self.num_classes = 10
            if self.usr_group is None:
                if self.iid:
                    self.usr_group = mnist_iid(
                        raw_dataset, self.num_partitions)
                else:
                    self.usr_group = mnist_noniid(
                        raw_dataset, self.num_partitions)
                for usr in self.usr_group.keys():
                    np.random.shuffle(self.usr_group[usr])
        elif self.name == 'cifar10':
            apply_transform = transforms.Compose(
                [transforms.ToTensor(),
                 transforms.Resize((self.width, self.height)),
                 transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
            raw_dataset = datasets.CIFAR10(self.data_dir, train=True, download=True,
                                           transform=apply_transform)
            self.num_classes = 10
            if self.usr_group is None:
                if self.iid:
                    self.usr_group = cifar_iid(
                        raw_dataset, self.num_partitions)
                else:
                    self.usr_group = cifar_noniid(raw_dataset, self.num_partitions,
                                                  shards_per_user=self.shards_per_partition)
                for usr in self.usr_group.keys():
                    np.random.shuffle(self.usr_group[usr])
        elif self.name == 'cifar100':
            apply_transform = transforms.Compose(
                [transforms.ToTensor(),
                 transforms.Resize((self.width, self.height)),
                 transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
            raw_dataset = datasets.CIFAR100(self.data_dir, train=True, download=True,
                                            transform=apply_transform)
            self.num_classes = 100
            if self.usr_group is None:
                if self.iid:
                    self.usr_group = cifar_iid(
                        raw_dataset, self.num_partitions)
                else:
                    self.usr_group = cifar_noniid(raw_dataset, self.num_partitions,
                                                  shards_per_user=self.shards_per_partition)
                for usr in self.usr_group.keys():
                    np.random.shuffle(self.usr_group[usr])
        elif self.name == 'tiny-imagenet':
            apply_transform = transforms.Compose(
                [transforms.ToTensor(),
                 transforms.Resize((self.width, self.height)),
                 transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
            raw_dataset = datasets.ImageFolder(
                self.data_dir, transform=apply_transform)
            self.num_classes = 1000
            if self.usr_group is None:
                if self.iid:
                    self.usr_group = tiny_imagenet_iid(
                        raw_dataset, self.num_partitions)
                else:
                    self.usr_group = tiny_imagenet_noniid(
                        raw_dataset, self.num_partitions)
                for usr in self.usr_group.keys():
                    np.random.shuffle(self.usr_group[usr])
        elif self.name == 'imagenet':
            # apply_transform = transforms.Compose(
            #     [transforms.ToTensor(),
            #      transforms.Resize((self.width, self.height)),
            #      transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
            # raw_dataset = datasets.ImageFolder(os.path.join(self.data_dir, 'train'), transform=apply_transform)
            raw_dataset = datasets.ImageFolder(
                os.path.join(self.data_dir, 'train'))
            self.num_classes = 1000
            if self.usr_group is None:
                if self.iid:
                    self.usr_group = imagenet_iid(
                        raw_dataset, self.num_partitions)
                else:
                    self.usr_group = imagenet_noniid(raw_dataset, self.num_partitions,
                                                     shards_per_user=self.shards_per_partition)
                for usr in self.usr_group.keys():
                    np.random.shuffle(self.usr_group[usr])
        else:
            raise NotImplementedError
        logging.info(f"Setup Data Partition{partition}")
        idx = self.usr_group[partition].astype(int)
        idx_train = idx[:int(self.train_split * len(idx))]
        idx_val = idx[len(idx_train):]
        if self.name != 'imagenet':
            self.train_dataset = DatasetSplit(raw_dataset, idx_train)
            self.val_dataset = DatasetSplit(raw_dataset, idx_val)
        else:
            train_transform = transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                                     0.229, 0.224, 0.225])
            ])
            val_transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                                     0.229, 0.224, 0.225])
            ])
            self.train_dataset = DatasetSplit(
                raw_dataset, idx_train, transform=train_transform)
            self.val_dataset = DatasetSplit(
                raw_dataset, idx_val, transform=val_transform)
        if update_usr_group:
            logging.warn(
                "Calculating KL divergence for new generated partition")
            class_cnts = np.zeros((self.num_partitions, self.num_classes))
            for p in range(self.num_partitions):
                idx = self.usr_group[p].astype(int)
                for iidx in tqdm(idx):
                    label = raw_dataset.targets[iidx]
                    class_cnts[p, label] += 1
            prob = class_cnts / np.sum(class_cnts, axis=1, keepdims=True)
            kl = np.sum(prob[1:] * np.log(prob[1:] /
                        (prob[0] + 1e-9) + 1e-9), axis=1)
            kl_str = "_".join([f"{k:.2f}" for k in kl])
            grp = np.array([np.concatenate([[k], v])
                           for k, v in self.usr_group.items()])
            save_path = os.path.join(self.data_dir, f"usr_group_{kl_str}.npy")
            np.save(save_path, grp)
            logging.warn(f"Partition def saved to {save_path}")
        logging.info(f"Setup Data Partition{partition}")

    def setup(self, stage):
        self.switch_partition(self.current_partition)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=self.train_shuffle,
                          num_workers=self.train_workers, pin_memory=False)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False,
                          num_workers=self.val_workers, pin_memory=False)

    def predict_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=1, shuffle=False,
                          num_workers=self.val_workers, pin_memory=True)

    def test_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=1, shuffle=False,
                          num_workers=self.val_workers, pin_memory=True)
