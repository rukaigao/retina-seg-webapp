import torch
from .attunet import AttUNetMultiHead  # import your model architecture

def load_model(path, device):
    model = AttUNetMultiHead(in_channels=3, base_channels=32, dropout=0.0).to(device)
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()
    return model

def segment_image(model, image_tensor):
    with torch.no_grad():
        output = model(image_tensor, task="A")
        return {
            "ava": torch.sigmoid(output["ava"][0])[0],
            "pnv": torch.sigmoid(output["pnv"][0])[0],
            "onh": torch.sigmoid(output["onh"])[0],
            "edges": torch.sigmoid(output["edges"])[0]
        }
