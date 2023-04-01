import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.optim as optim
from PIL import Image
import wandb

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")
torch.manual_seed(100)

path_to_data = "C:\Users\Utilisateur\Desktop\Etudes\Liège\Cours\Q4\Deep_learning\project\images_gz2\images\"

# Hyperparameters
batch_size = 32
block_size = 512
n_embedding = 512
n_blocks = 12
n_head = 8        
dropout = 0.2
max_iters = 50000
learning_rate = 3e-4
eval_every = 500
eval_iters = 200
save_every = 10000

#hyperparameters of the NN themselves
input_features =3*32*32 # RGB pixel (dim=3)* 32*32 image
output_features = 1 #
hidden_features = 64 #random for now
learning_rate = 0.005 # random for now
num_epochs = 3 # random for now

model_name = "vae-project-DL"
directory = ""
checkpoint = f"{directory}/{model_name}.pt"

# WandB
wandb.init(
    project="vae",
    config={
        "model": model_name,
        "batch_size": batch_size,
        "block_size": block_size,
        "n_embedding": n_embedding,
        "n_blocks": n_blocks,
        "n_head": n_head,
        "dropout": dropout,
        "max_iters": max_iters,
        "learning_rate": learning_rate,
        "eval_every": eval_every,
        "eval_iters": eval_iters,
    }
)


class VAEModel(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self):

        return
    
    def loss(self):

        return
    
    def generate(self):

        return
    
model = VAEModel()
model=model.to(device)