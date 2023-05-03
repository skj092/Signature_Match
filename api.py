import falcon
import json
import torch
from PIL import Image
from io import BytesIO
from src.utils import transform, predict_single
from pathlib import Path
import timm
from fastai.vision.all import create_body, create_head, Module

class SiameseModel(Module):
    def __init__(self, encoder, head):
        self.encoder,self.head = encoder,head

    def forward(self, x1, x2):
        ftrs = torch.cat([self.encoder(x1), self.encoder(x2)], dim=1)
        return self.head(ftrs)

def load_model(model_path):
    resnet18 = timm.create_model('resnet18', pretrained=True)
    encoder = create_body(resnet18, cut=-2)
    head = create_head(512*2, 2, ps=0.5)
    model = SiameseModel(encoder, head)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')), strict=False)
    model.eval()
    return model

class ImageClassifier:
    def __init__(self):
        self.model = load_model('models/siamese_model.pth')
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
    
    def validate_json(self, req):
        if req.content_type != 'application/json':
            raise falcon.HTTPBadRequest(
                'Invalid JSON',
                'A valid JSON document is required.'
            )
        else :
            return True

    def process_image(self, img1, img2):
        img1 = Image.open(img1).convert('RGB')
        img2 = Image.open(img2).convert('RGB')
        output, prob = predict_single(self.model, img1, img2)
        return output, prob

    def on_post(self, req, resp):
        resp.status = falcon.HTTP_200
        validate = self.validate_json(req)
        if validate:
            inp = req.stream.read().decode('utf-8')
            inp = json.loads(inp)
            img1, img2 = inp['img1'], inp['img2']
            output, prob = self.process_image(img1, img2)
            prob = round(prob, 2)
            print(f"output: {output}, prob: {prob}")
            prediction = "Matching" if output == 1 else "Not Matching"
            resp.body = json.dumps({'output': prediction, 'probability': prob})


            



app = falcon.App()
app.add_route('/classify', ImageClassifier())
