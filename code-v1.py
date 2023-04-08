import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.optim as optim
import torch.utils.data as data
from torchvision import datasets, transforms, utils
import matplotlib.pyplot as plt
from PIL import Image
import wandb
import os
from glob import glob

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


device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")
torch.manual_seed(100)

path_to_data = "\images_gz2"


# Hyperparameters
batch_size = 32
n_embedding = 512
dropout = 0.2
max_iters = 50000
learning_rate = 3e-4
eval_every = 500
eval_iters = 200
save_every = 10000

#hyperparameters of the NN specifically
input_features =3*32*32 # RGB pixel (dim=3)* 32*32 image
output_features = 128 # size of output of encoder = input of decoder
hidden_features = 256 # arbitrary
latent_space_size = 2 #size of the latent space --> arbitrary
learning_rate = 0.005 # arbitrary
num_epochs = 10 # arbitrary

model_name = "vae-project-DL"
directory = ""
checkpoint = f"{directory}/{model_name}.pt"

# WandB
wandb.init(
    project="vae",
    config={
        "model": model_name,
        "batch_size": batch_size,
        "n_embedding": n_embedding,
        "max_iters": max_iters,
        "learning_rate": learning_rate,
        "eval_every": eval_every,
        "eval_iters": eval_iters,
        "dropout": dropout,
    }
)

DataURL='https://tinyurl.com/mr2yc5nx'

RetrieveData(DataURL)

image_path='./data/images_gz2/images'

class GalaxyZoo2(Dataset):
	def __init__(self, img_dir, transform=None, train=True):
		self.transform = transform
		self.img_dir = img_dir
		
		images=glob(os.path.join(img_dir,'*.jpg'))
		rd.shuffle(images)
		
		cut=int(0.6*len(images))
		if train:
			self.images=images[:cut]
		else:
			self.images=images[cut:]
		
	def __len__(self):
		return len(self.images)
	
	def __getitem__(self, index):
		img_to_tensor=transforms.Compose([transforms.PILToTensor()])
		img_path=self.images[index]
		with Image.open(img_path) as im:
			im=img_to_tensor(im)
			if self.transform:
				im = self.transform(im)
			return im.float()


class VAEModel(nn.Module):
    def __init__(self,latent_space_size):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_features, hidden_features),
            nn.ReLU(),
            nn.Linear(hidden_features, hidden_features),
            nn.ReLU(),
            #maybe add a batch normalization (BatchNorm2d and then ReLU) here
            nn.Linear(hidden_features, output_features),
        )

        #make the losses for later
        self.reconstructloss = 0
        self.KL = 0
        #make the latent space parameters
        self.epsilon = torch.normal(0,1)
        self.mu = nn.Linear(hidden_features,latent_space_size)
        self.logvar = nn.Linear(hidden_features,latent_space_size)
        self.latent = nn.Linear(latent_space_size,hidden_features)

        self.decoder = nn.Sequential(
            nn.Linear(hidden_features, hidden_features),
            nn.ReLU(),
            nn.Linear(hidden_features, hidden_features),
            nn.ReLU(),
            #maybe add a batch normalization (BatchNorm2d and then ReLU) here
            nn.Linear(hidden_features, input_features),
            nn.Softmax(), #maybe try smth else if softmax doesn't work well (like sigmoid f.i.)
        )

        
    def forward(self, x):
        x.to(device)
        encoded = self.encoder(x)
        #create the latent space by taking the mean and log of variance to make a normal distribution to pick in
        mu = self.mu(encoded)
        epsi = self.epsilon.sample(len(mu)) #reparametrization trick (make a sample of the size of mu)
        sigma = torch.exp(0.5 * self.logvar(encoded))
        z = mu + sigma * epsi
        lat = self.latent(z) #make the latent space
        decoded = self.decoder(lat)
        return decoded


def elboloss(self, x_gen, x_in):
    """
    Compute the ELBO loss function of the VAE
    x_gen = generated output from the decoder of the VAE
    x_in = target input to compare the generated one with
    """
    criterion = F.binary_cross_entropy(x_gen, x_in) #reconstruction loss by BCE
    KLdivloss = 0.5*torch.sum(1 + self.logvar - self.mu**2 - torch.exp(self.logvar)) #regularization loss by KL

    elbo = torch.mean(criterion - KLdivloss)
    return elbo
    
network = VAEModel(latent_space_size=2)
network = network.to(device)

optimizer = optim.Adam() #could be another one than Adam btw
criterion = elboloss()


trainset=GalaxyZoo2(image_path)
trainloader=DataLoader(trainset, batch_size=4, shuffle=True, num_workers=2)

images=next(iter(trainloader))
show_images(utils.make_grid(images))

def train(num_epochs):
    """
    Train the model (VAE) onto the data in the trainloader
    """
    train_avg_loss = []

    for i in range(num_epochs):
        print(i) #just to see it advance (check if not stuck and time needed)
        train_losses = []
        
        for inputs in trainloader:
            inputs = inputs.to(device)
            
            pred = network(inputs)
            loss = criterion(pred,inputs) #compute the ELBO loss
            train_losses.append(loss.detach())
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
        train_avg_loss.append(torch.mean(torch.FloatTensor(train_losses)))

    return train_avg_loss #we could also return the reconstruction loss and the regularization loss individually, but in the end it is the ELBO loss that is important
train_avg_loss = train(num_epochs)

fig = plt.figure()

plt.plot(train_avg_loss)
plt.title('ELBO loss of the VAE')
plt.xlabel('Iterations')
plt.ylabel('Loss (ELBO)')

plt.show()

print('a')