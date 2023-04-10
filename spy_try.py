import os
from glob import glob
from utils.custom_utils import RetrieveData, show_images
import pandas as pd
import random as rd

from PIL import Image
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from torchvision.io import read_image

DataURL='https://tinyurl.com/mr2yc5nx'

RetrieveData(DataURL)

image_path='./data/images_gz2/images/'

class GalaxyZoo2(Dataset):
    def __init__(self, img_dir, transform=None, train=True):
        self.transform = transform
        self.img_dir = img_dir
        
        images=glob(os.path.join(img_dir,'*.jpg'))
        rd.shuffle(images)
        
    
        cut=int(0.1*len(images)/100)
        if train:
            self.images = images[:cut]
        else:
            self.images = images[cut:]
    
    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        
        img_path=self.images[index]
        image = read_image(img_path)/255
        image = image.float()
        if self.transform:
            image = self.transform(image)
        return image

#trainset=GalaxyZoo2(image_path)

#trainloader=DataLoader(trainset, batch_size=4, shuffle=True, num_workers=2)

#images=next(iter(trainloader))
#show_images(utils.make_grid(trainset[0]))


class MLP_VAE(nn.Module):
    
    def __init__(self):
        self.img_dim = 3*424*424
        super().__init__()        
        self.encoder = torch.nn.Sequential(
            torch.nn.Flatten(0),
            torch.nn.Linear(self.img_dim, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 10)
        )
          
        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(10, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, self.img_dim),
            torch.nn.Unflatten(0,(3,424,424))
        )
        
        
    def sample_z(self, mu, logvar):
        eps = torch.rand_like(mu)
        return mu + eps * torch.exp(0.5 * logvar)
        
    def forward(self, x):
        img_to_tensor = transforms.Compose([transforms.PILToTensor()])
        #x = img_to_tensor(x)
        #x = torch.reshape(x,(1,self.img_dim))
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

class MLP_Discriminator(nn.Module):
    
    def __init__(self):
        self.img_dim = 3*424*424
        super().__init__()        
        self.encoder = torch.nn.Sequential(
            torch.nn.Flatten(0),
            torch.nn.Linear(self.img_dim, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 10),
            torch.nn.ReLU(),
            torch.nn.Linear(10, 1),
            torch.nn.Sigmoid()
        )
        
    def forward(self, x):
        img_to_tensor = transforms.Compose([transforms.PILToTensor()])
        #x = img_to_tensor(x)
        #x = torch.reshape(x,(1,self.img_dim))
        encoded = self.encoder(x)
        return encoded

network = MLP_VAE()
disc = MLP_Discriminator()
learning_rate = 0.01
#criterion = nn.BCELoss()
criterion = nn.L1Loss()
optimizer_disc = torch.optim.Adam(network.parameters(), lr=learning_rate)
optimizer_net = torch.optim.Adam(network.parameters(), lr=learning_rate)

transform = transforms.Compose([
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ColorJitter()
])


'''
class CONV_VAE(nn.Module):
    
    def __init__(self):
        self.img_dim = 3*424*424
        super().__init__()        
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(self.img_dim, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 10)
        )
          
        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(10, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, self.img_dim)
        )
        
        
    def sample_z(self, mu, logvar):
        eps = torch.rand_like(mu)
        return mu + eps * torch.exp(0.5 * logvar)
        
    def forward(self, x):
        img_to_tensor = transforms.Compose([transforms.PILToTensor()])
        #x = img_to_tensor(x)        #x = torch.reshape(x,(1,self.img_dim))
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        decoded = torch.reshape(decoded,(3,424,424))
        return decoded

class CONV_Discriminator(nn.Module):
    
    def __init__(self):
        self.img_dim = 3*424*424
        super().__init__()        
        self.encoder = torch.nn.Sequential(
            torch.nn.Dropout(0.1),
            torch.nn.Conv2d(self.img_dim,212,(424,212)),
            torch.nn.ReLU(),
            torch.nn.Linear(212,128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 10),
            torch.nn.ReLU(),
            torch.nn.Linear(10, 1),
            torch.nn.Sigmoid(),
        )
        
    def forward(self, x):
        img_to_tensor = transforms.Compose([transforms.PILToTensor()])
        #x = img_to_tensor(x)
        x = torch.reshape(x,(1,self.img_dim))
        encoded = self.encoder(x)
        return encoded

network = CONV_VAE()
disc = CONV_Discriminator()
learning_rate = 0.01
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(network.parameters(), lr=learning_rate)
'''


transform = transforms.Compose([
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ColorJitter()
])


trainset=GalaxyZoo2(image_path,transform=transform)
print(len(trainset))
#trainset = trainset[:50]

device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def train_disc(num_epochs):
    train_losses_true = []
    train_losses_false = []

    for i in range(num_epochs):
        print(i)
        
        
        for y_true in trainset:
            pred = disc(y_true)
            loss = criterion(torch.tensor([0.]), pred)
            train_losses_true.append(loss.detach())
            
            optimizer_disc.zero_grad()
            loss.backward()
            optimizer_disc.step()
            
            y_false = torch.rand((3, 424,424))*2 -1
            pred = disc(y_false)
            loss = criterion(torch.tensor([1.]), pred)
            train_losses_false.append(loss.detach())
            
            optimizer_disc.zero_grad()
            loss.backward()
            optimizer_disc.step()

    return train_losses_true, train_losses_false


def train_gen(num_epochs):
    train_loss = []
    for i in range(num_epochs):
        print(i)
        pred = network(torch.rand((3,424,424))*2 -1)
        guess = disc(pred)
        loss = criterion(torch.tensor([0.]), guess)
        
        optimizer_net.zero_grad()
        loss.backward()
        optimizer_net.step()
        
        train_loss.append(loss.detach())
    
    return train_loss


#pred = network(trainset[0])


a,b = train_disc(1)
c = train_gen(20)

#inv_tensor = trainset[0]*0.5 + 0.5
#show_images(utils.make_grid(inv_tensor))


img = network(torch.rand((3,424,424))*2 -1)
img_denormalized = img*0.5 + 0.5
show_images(utils.make_grid(img_denormalized))
plt.show()

#show_images(utils.make_grid(trainset[0]))