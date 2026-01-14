import torch
import io
import os
from flask import Flask, render_template, request, jsonify
from PIL import Image
import torchvision.transforms as transforms
from model import MNIST_model_54

app = Flask(__name__)

# Load Model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MNIST_model_54()
model_path = "handwtitten_number_detector_model.pth"

if os.path.exists(model_path):
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval()
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Error loading model weights: {e}")
else:
    print(f"Warning: {model_path} not found. Please move your .pth file into this folder.")

def transform_image(image_bytes):
    img = Image.open(io.BytesIO(image_bytes)).convert("L").resize((28, 28))
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    return transform(img).unsqueeze(0).to(device)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file part"})
    file = request.files["file"]
    img_bytes = file.read()
    tensor = transform_image(img_bytes)
    with torch.no_grad():
        outputs = model(tensor)
        _, predicted = torch.max(outputs, 1)
    return jsonify({"prediction": int(predicted.item())})

if __name__ == "__main__":
    app.run(debug=True, port=5000)
