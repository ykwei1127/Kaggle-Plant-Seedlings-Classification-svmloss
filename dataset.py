from torch.utils.data import DataLoader
from torchvision import datasets
from torch.utils.data import Dataset
from torch.utils.data.sampler import SubsetRandomSampler
import numpy as np

valid_size = 0.2

def make_train_data_loader(dir, transforms):

    trainset = datasets.ImageFolder(dir, transform=transforms)

    num_train = len(trainset)
    indices = list(range(num_train))
    split = int(np.floor(valid_size * num_train))
    np.random.shuffle(indices)

    train_idx, valid_idx = indices[split:], indices[:split]
    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)

    train_data_loader = DataLoader(dataset=trainset, batch_size=32, num_workers=4, sampler=train_sampler)
    valid_data_loader = DataLoader(dataset=trainset, batch_size=32, num_workers=4, sampler=valid_sampler)
    # print(train_data_loader.dataset.imgs)
    # print(len(train_data_loader.dataset.imgs))
    
    return train_data_loader, valid_data_loader
