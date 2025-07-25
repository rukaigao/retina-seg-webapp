from flask import Flask, request, render_template, send_from_directory
import os
from werkzeug.utils import secure_filename
from model.model import load_model, segment_image
from utils import load_image, load_mask, dice_score, save_mask
import torch

app = Flask(__name__)
UPLOAD_FOLDER = "static/output"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load model once
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = load_model("model/multitask_model.pth", device)

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        raw_file = request.files["raw"]
        masks = {
            "ava": request.files.get("ava"),
            "pnv": request.files.get("pnv"),
            "onh": request.files.get("onh"),
            "edges": request.files.get("edges")
        }

        filename = secure_filename(raw_file.filename)
        raw_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        raw_file.save(raw_path)

        image_tensor = load_image(raw_path, device)
        pred_masks = segment_image(model, image_tensor)

        dice_scores = {}
        for key, file in masks.items():
            pred = pred_masks[key]
            pred_path = os.path.join(app.config["UPLOAD_FOLDER"], f"{key}_pred.png")
            save_mask(pred, pred_path)

            if file:
                gt_path = os.path.join(app.config["UPLOAD_FOLDER"], f"{key}_gt.png")
                file.save(gt_path)
                gt_mask = load_mask(gt_path, device)
                dice_scores[key] = dice_score(pred, gt_mask).item()
            else:
                dice_scores[key] = None

        return render_template("index.html", dice=dice_scores, filenames=pred_masks.keys())

    return render_template("index.html", dice=None)

@app.route('/static/output/<filename>')
def download_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename, as_attachment=True)

# if __name__ == "__main__":
    # app.run(debug=True)
