import matplotlib.pyplot as plt
import torch
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
from torchvision import transforms

def visualize_batch(batch):
    image1_batch, image2_batch, label_batch = batch

    for i in range(len(image1_batch)):
        image1 = image1_batch[i]
        image2 = image2_batch[i]
        label = label_batch[i]
        label = 'Original' if label.item() ==0 else 'Forged'

        fig, (ax1, ax2) = plt.subplots(1, 2)
        fig.suptitle('Label: {}'.format(label))

        # Display image 1
        ax1.imshow(image1.permute(1, 2, 0), cmap='gray')
        ax1.set_title('Image 1')

        # Display image 2
        ax2.imshow(image2.permute(1, 2, 0), cmap='gray')
        ax2.set_title('Image 2')

        plt.show()

def predict(model, dl):
    preds = []
    for x1, x2, y in tqdm(dl):
        preds.append(model(x1, x2).argmax(dim=1))
    return torch.cat(preds)

def plot_confusion_matrics(model, dl):
    preds = predict(model, dl)
    actual = torch.cat([y for x1, x2, y in dl])
    cm = confusion_matrix(actual, preds)
    return cm

# transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485],std=[0.229])
])

def predict_single(model, img1, img2):
    '''Input: model, PIL image, PIL image
       Output: label, probability'''
    img1 = transform(img1)
    img2 = transform(img2)
    img1 = img1.unsqueeze(0)
    img2 = img2.unsqueeze(0)
    out = model(img1, img2)
    out = torch.softmax(out, dim=1)
    label, prob = out.argmax().item(), out.max().item()
    return label, prob
