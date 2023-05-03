# importing necessary libraries
from torch.utils.data import Dataset, DataLoader, Subset, random_split
from pathlib import Path
from PIL import Image
import os
from torchvision import transforms
import matplotlib.pyplot as plt
from torch import nn, optim
from torch.nn import functional as F
import torch
from torchvision import models
import pandas as pd
from tqdm import tqdm
from models import SiameseNetwork


# config
class Config:
    kaggle = False
    if kaggle:
        path = Path('/kaggle/input/re-arranged-data/sign_data/')
    else:
        path = Path('../sign_data')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch_size = 32
    num_epochs = 10
    lr=0.001
    epochs = 10


cfg = Config()

# Dataset class
class SiameseDataset(Dataset):
    def __init__(self, df, root_dir=cfg.path/'Dataset/train', transform=None):
        self.df = df
        self.root_dir = root_dir
        self.transform = transform
    def __getitem__(self, index):
        row = self.df.iloc[index]
        img1 = Image.open(os.path.join(self.root_dir, row['img1']))
        img2 = Image.open(os.path.join(self.root_dir, row['img2']))
        # convert to grayscale
        # img1 = img1.convert('L')
        # img2 = img2.convert('L')
        label = row['target']
        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
        return img1, img2, label

    def __len__(self):
        return len(self.df)

# transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485],std=[0.229])
])

def train(model, train_dl, valid_dl, optimizer, epochs):
    train_loss, valid_loss = [], []
    train_acc, valid_acc = [], []
    # training loop
    for epoch in range(epochs):
        # training
        model.train()
        for batch in tqdm(train_dl):
            img1, img2, label = batch
            img1 = img1.to(cfg.device)
            img2 = img2.to(cfg.device)
            label = label.to(cfg.device)
            optimizer.zero_grad()
            loss = model.get_loss(img1, img2, label)
            loss.backward()
            optimizer.step()
        train_loss.append(loss.item())
        train_acc.append(model.get_accuracy(img1, img2, label))
        # validation
        model.eval()
        with torch.no_grad():
            for batch in tqdm(valid_dl):
                img1, img2, label = batch
                img1 = img1.to(cfg.device)
                img2 = img2.to(cfg.device)
                label = label.to(cfg.device)
                loss = model.get_loss(img1, img2, label)
        valid_loss.append(loss.item())
        valid_acc.append(model.get_accuracy(img1, img2, label))
        print('Epoch: {} Train Loss: {:.4f} Train Accuracy: {:.4f} Valid Loss: {:.4f} Valid Accuracy: {:.4f}'\
              .format(epoch, sum(train_loss)/len(train_loss), sum(train_acc)/len(train_acc), sum(valid_loss)/len(valid_loss), sum(valid_acc)/len(valid_acc)))
    return train_loss, valid_loss, train_acc, valid_acc


if __name__=='__main__':
    df = pd.read_csv(cfg.path/'train_data.csv')
    df = df.sample(1000)

    # splitting the dataset such that "068/09_068.png" string before "/" is different for train and val
    df["unique"] = df["img1"].apply(lambda x: x.split('/')[0])

    train_unique = df["unique"].unique()[:int(len(df["unique"].unique())*0.8)]
    val_unique = df["unique"].unique()[int(len(df["unique"].unique())*0.8):]

    # split the dataset by different "unique" column
    train_df = df[df["unique"].isin(train_unique)]
    valid_df = df[df["unique"].isin(val_unique)]

    train_dataset = SiameseDataset(train_df, transform=transform)
    valid_dataset = SiameseDataset(valid_df, transform=transform)

    train_dl = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True)
    valid_dl = DataLoader(valid_dataset, batch_size=cfg.batch_size, shuffle=True)

    model = SiameseNetwork().to(cfg.device)
    optimizer = optim.Adam(model.parameters(), lr=cfg.lr)

    # train model
    train_loss, valid_loss, train_acc, valid_acc = train(model, train_dl, valid_dl, optimizer, cfg.epochs)

