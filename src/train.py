from fastai.vision.all import *
from torch.utils.data import Dataset, DataLoader, Subset, random_split
from torchvision import transforms
import timm
from sklearn.model_selection import train_test_split
from pathlib import Path
from sklearn.metrics import confusion_matrix
from src.utils import predict, plot_confusion_matrics

# For reproducibility
torch.manual_seed(0)
np.random.seed(0)

# config
class cfg:
    kaggle = False
    if kaggle:
        path = Path('/kaggle/input/re-arranged-data/sign_data/')
    else:
        path = Path('../sign_data')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch_size = 32
    training_model = True

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


df = pd.read_csv(cfg.path/'train_data.csv')
df = df.sample(1000)

# splitting the dataset such that "068/09_068.png" string before "/" is different for train and val
df["unique"] = df["img1"].apply(lambda x: x.split('/')[0])

train_unique = df["unique"].unique()[:int(len(df["unique"].unique())*0.8)]
val_unique = df["unique"].unique()[int(len(df["unique"].unique())*0.8):int(len(df["unique"].unique())*0.9)]
test_unique = df["unique"].unique()[int(len(df["unique"].unique())*0.9):]

# split the dataset by different "unique" column
train_df = df[df["unique"].isin(train_unique)]
valid_df = df[df["unique"].isin(val_unique)]
test_df = df[df["unique"].isin(test_unique)]
print(f"train_df: {len(train_df)}, valid_df: {len(valid_df)}, test_df: {len(test_df)}")

train_dataset = SiameseDataset(train_df, transform=transform)
valid_dataset = SiameseDataset(valid_df, transform=transform)
test_dataset = SiameseDataset(test_df, transform=transform)

train_dl = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True)
valid_dl = DataLoader(valid_dataset, batch_size=cfg.batch_size, shuffle=True)
test_dl = DataLoader(test_dataset, batch_size=cfg.batch_size, shuffle=True)

# create fastai dataloader
dls = DataLoaders(train_dl, valid_dl)

# learner
class SiameseModel(Module):
    def __init__(self, encoder, head):
        self.encoder,self.head = encoder,head

    def forward(self, x1, x2):
        ftrs = torch.cat([self.encoder(x1), self.encoder(x2)], dim=1)
        return self.head(ftrs)

resnet34 = timm.create_model('resnet18', pretrained=True)
encoder = create_body(resnet34, cut=-2)
head = create_head(512*2, 2, ps=0.5)
model = SiameseModel(encoder, head)

def siamese_splitter(model):
    return [params(model.encoder), params(model.head)]


learn = Learner(dls, model, splitter=siamese_splitter, loss_func=CrossEntropyLossFlat(), metrics=accuracy)

# save model and loading
print('saving model')
if cfg.training_model:
    learn.fine_tune(3)
    learn.export('siamese_model')
else:
    learn.load('siamese_model')

# save model and loading
# if cfg.training_model:
#     print('Training model')
#     learn.fine_tune(3)
#     learn.save('../siamese_model', with_opt=False)
# else:
#     print('loading model')
#     learn = SiameseModel(encoder, head)
#     model = torch.load('../models/siamese_model.pth', map_location=cfg.device)
#     learn.load_state_dict(model)

# # plot confusion matrix on train dl
# print(plot_confusion_matrics(learn, train_dl))

# # plot confusion matrix on valid dl
# print(plot_confusion_matrics(learn, valid_dl))

# # plot confusion matrix on test dl
# print(plot_confusion_matrics(learn, test_dl))



# # plot confusion matrix on train dl and sklearn
# preds, targs = learn.get_preds(dl=train_dl)
# preds = preds.argmax(dim=-1)
# cm = confusion_matrix(targs, preds)
# print(cm)

# plot confusion matrix on valid dl and sklearn
preds, targs = learn.get_preds(dl=valid_dl)
preds = preds.argmax(dim=-1)
cm = confusion_matrix(targs, preds)
print(cm)

# plot confusion matrix on test dl and sklearn
preds, targs = learn.get_preds(dl=test_dl)
preds = preds.argmax(dim=-1)
cm = confusion_matrix(targs, preds)
print(cm)
