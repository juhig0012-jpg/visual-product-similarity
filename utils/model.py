import torch
from torchvision import models, transforms
from sklearn.preprocessing import normalize
from PIL import Image
import io

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = models.resnet50(weights='IMAGENET1K_V1').to(device)
model.eval()

preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225]),
])

def extract_embedding(image_path_or_bytes):
    img = Image.open(image_path_or_bytes).convert('RGB')
    img_t = preprocess(img).unsqueeze(0).to(device)
    with torch.no_grad():
        emb = model(img_t)[0].cpu().numpy()
    return normalize(emb.reshape(1,-1))[0]