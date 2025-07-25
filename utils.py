from PIL import Image
import numpy as np
import torch
import torchvision.transforms as T

def load_image(path, device):
    img = Image.open(path).convert("RGB").resize((1024, 1024))
    tensor = T.ToTensor()(img).unsqueeze(0).to(device)
    return tensor

def load_mask(path, device):
    mask = Image.open(path).convert("L").resize((1024, 1024))
    tensor = T.ToTensor()(mask).to(device)
    return (tensor > 0.5).float()

def save_mask(tensor, path):
    arr = (tensor.squeeze().cpu().numpy() > 0.5).astype(np.uint8) * 255
    Image.fromarray(arr).save(path)

def dice_score(pred, target, smooth=1e-6):
    pred = (pred > 0.5).float().view(-1)
    target = target.view(-1)
    intersection = (pred * target).sum()
    return (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)
