from torch.utils.data import DataLoader, Dataset
from PIL import Image
import torch
from albumentations import *
from albumentations.pytorch import ToTensorV2
from torch.utils.data.sampler import SubsetRandomSampler
import glob
import numpy as np


class train_i1_transforms:
    def __init__(self):
        self.i1_transform = Compose([
            Resize(112, 112),
            # Normalize(mean=[0.4914, 0.48216, 0.44653], std=[0.24703, 0.24349, 0.26159]),
            ToTensorV2(),
        ])

    def __call__(self, img):
        img = np.array(img)
        img = self.i1_transform(image=img)['image']
        return img


class train_i2_transforms:
    def __init__(self):
        self.i2_transform = Compose([
            Resize(112, 112),
            # Normalize(mean=[0.4914, 0.48216, 0.44653], std=[0.24703, 0.24349, 0.26159]),
            ToTensorV2(),
        ])

    def __call__(self, img):
        img = np.array(img)
        img = self.i2_transform(image=img)['image']
        return img


class test_i1_transforms:
    def __init__(self):
        self.i1_transform = Compose([
            Resize(112, 112),
            # Normalize(mean=[0.4914, 0.48216, 0.44653], std=[0.24703, 0.24349, 0.26159]),
            ToTensorV2(),
        ])

    def __call__(self, img):
        img = np.array(img)
        img = self.i1_transform(image=img)['image']
        return img


class test_i2_transforms:
    def __init__(self):
        self.i2_transform = Compose([
            Resize(112, 112),
            # Normalize(mean=[0.4914, 0.48216, 0.44653], std=[0.24703, 0.24349, 0.26159]),
            ToTensorV2(),
        ])

    def __call__(self, img):
        img = np.array(img)
        img = self.i2_transform(image=img)['image']
        return img


class o1_transforms:
    def __init__(self):
        self.o1_transform = Compose([
            Resize(112, 112),
            # Normalize(mean=[0.4914, 0.48216, 0.44653], std=[0.24703, 0.24349, 0.26159]),
            ToTensorV2(),
        ])

    def __call__(self, img):
        img = np.array(img)
        img = img[:, :, np.newaxis]
        img = self.o1_transform(image=img)['image']
        return img


class CustomDataset(Dataset):
    def __init__(self, transform_input1=None, transform_input2=None, transform_output1=None, root='./data/',
                 valid=False):
        self.i1_paths = sorted(glob.glob(root + 'bg/*'))
        self.i2_paths = sorted(glob.glob(root + 'fg_bg/*/*', recursive=True))
        self.o1_paths = sorted(glob.glob(root + 'fg_bg_mask/*/*', recursive=True))
        self.transform_i1 = transform_input1
        self.transform_i2 = transform_input2
        self.transform_o1 = transform_output1
        self.valid = valid

    def __len__(self):
        if self.valid:
            return int(0.3 * len(self.i2_paths))
        return len(self.i2_paths) - int(0.3 * len(self.i2_paths))

    def __getitem__(self, index):
        i1_index = int(index / 4000)  # Since same copy of i1 is required for 4000 data items
        i1 = Image.open(self.i1_paths[i1_index])
        i2 = Image.open(self.i2_paths[index])
        o1 = Image.open(self.o1_paths[index])

        if self.transform_i1:
            i1 = self.transform_i1(i1)
        if self.transform_i2:
            i2 = self.transform_i1(i2)
        if self.transform_o1:
            o1 = self.transform_o1(o1)

        return {'i1': i1, 'i2': i2, 'o1': o1}


def getdata(root='./data/', batch_size=16, num_workers=0):
    train_dataset = CustomDataset(transform_input1=train_i1_transforms(), transform_input2=train_i2_transforms(),
                                  transform_output1=o1_transforms(), root=root)
    valid_dataset = CustomDataset(transform_input1=test_i1_transforms(), transform_input2=test_i2_transforms(),
                                  transform_output1=o1_transforms(), root=root, valid=True)

    validation_split = 0.3
    shuffle_dataset = True
    random_seed = 42

    # Creating data indices for training and validation splits:
    dataset_size = len(train_dataset) + len(valid_dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(validation_split * dataset_size))
    if shuffle_dataset:
        np.random.seed(random_seed)
        np.random.shuffle(indices)
    train_idx, valid_idx = indices[split:], indices[:split]

    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)

    SEED = 1
    cuda = torch.cuda.is_available()  # CUDA?
    print("CUDA Available?", cuda)

    # For reproducibility
    torch.manual_seed(SEED)
    if cuda:
        torch.cuda.manual_seed(SEED)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler, num_workers=num_workers,
                              pin_memory=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, sampler=valid_sampler, num_workers=num_workers,
                              pin_memory=True)

    return train_loader, valid_loader
