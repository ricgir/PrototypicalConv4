import os.path as osp
from PIL import Image

import torch
from torch.utils.data import Dataset
from torchvision import datasets, transforms

ROOT_PATH = './data/'

class CIFAR100(Dataset):

    def __init__(self, setname):
       
        if setname == 'train':
            train_split = True
        else:
            train_split = False

        cifar_dataset = datasets.CIFAR100(
            root=ROOT_PATH,
            train=train_split,
            download=True
        )
        
        self.data = cifar_dataset.data   
        self.label = cifar_dataset.targets 
        
        self.transform = transforms.Compose([
            transforms.Resize(84), # Resize to 84x84
            transforms.CenterCrop(84),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5071, 0.4867, 0.4408],
                                 std=[0.2675, 0.2565, 0.2761])
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        image_data, label = self.data[i], self.label[i]
        
        image = Image.fromarray(image_data)
        
        image = self.transform(image)
        return image, label

if __name__ == '__main__':
    trainset = CIFAR100(setname='train')
    print('Train set size:', len(trainset))
    testset = CIFAR100(setname='test')
    print('Test set size:', len(testset))
    valset = CIFAR100(setname='val')
    print('Val set size:', len(valset))
    
    img, lbl = trainset[0]
    print('Image shape:', img.shape)