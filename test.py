import torch
import torch.nn as nn
import timm
from fastai.vision.all import create_body, create_head, Module
from src.utils import predict_single, transform
from pathlib import Path
import pandas as pd
from PIL import Image
import os

# config
class cfg:
    path = Path('sign_data')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class SiameseModel(Module):
    def __init__(self, encoder, head):
        self.encoder,self.head = encoder,head

    def forward(self, x1, x2):
        ftrs = torch.cat([self.encoder(x1), self.encoder(x2)], dim=1)
        return self.head(ftrs)

def load_model(model_path):
    resnet34 = timm.create_model('resnet18', pretrained=True)
    encoder = create_body(resnet34, cut=-2)
    head = create_head(512*2, 2, ps=0.5)
    model = SiameseModel(encoder, head)
    model.load_state_dict(torch.load(model_path, map_location=cfg.device), strict=False)
    model.eval()
    return model


if __name__ == '__main__':
    image_dir = os.path.join(cfg.path, "Dataset/train")
    train_df = pd.read_csv(cfg.path/'train_data.csv')
    model_path = "models/siamese_model.pth"
    print(f"train_df: {len(train_df)}")
    print(f"Loading model")
    model = load_model(model_path)
    print(f"Model loaded")
    print(f"Predicting")
    for i in range(100):
        image1 = train_df.iloc[i]['img1']
        image2 = train_df.iloc[i]['img2']
        label = train_df.iloc[i]['target']

        image1 = Image.open(os.path.join(image_dir, image1))
        image2 = Image.open(os.path.join(image_dir, image2))
        lbl, prob = predict_single(model, image1, image2)
        print(f"label: {label}, predicted label: {lbl}, prob: {prob}")



