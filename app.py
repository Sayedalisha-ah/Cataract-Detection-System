import os
import torch
import torch.nn as nn
from torchvision import transforms
import timm
from PIL import Image
from flask import Flask, request, render_template, redirect, url_for, flash

# Initialize Flask app and configure upload folder
app = Flask(__name__)
app.secret_key = "your_secret_key"  # required for flash messages
UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Set device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Define data transformations (should match training preprocessing)
data_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],  # ImageNet statistics
                         std=[0.229, 0.224, 0.225])
])

def predict_image(model, image_path, transform, class_names):
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0)  # Add batch dimension
    image_tensor = image_tensor.to(device)
    model.eval()
    with torch.no_grad():
        output = model(image_tensor)
        _, predicted = torch.max(output, 1)
    return class_names[predicted.item()]

def load_model(model_path, class_names):
    num_classes = len(class_names)
    # Rebuild model architecture
    model = timm.create_model('vit_base_patch16_224', pretrained=False)
    model.head = nn.Linear(model.head.in_features, num_classes)
    model = model.to(device)

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model weights not found at {model_path}.")

    model.load_state_dict(torch.load(model_path, map_location=device))
    return model

# Load the model once when the app starts
MODEL_PATH = 'vit_cataract_model.pth'
CLASS_NAMES = ['CATARACT', 'NORMAL']
model = load_model(MODEL_PATH, CLASS_NAMES)
print("Model loaded successfully.")

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    filename = None
    if request.method == 'POST':
        if 'image' not in request.files:
            flash("No file part in the request.")
            return redirect(request.url)
        file = request.files['image']
        if file.filename == '':
            flash("No file selected.")
            return redirect(request.url)
        if file:
            filename = file.filename
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            # Perform prediction on the saved file
            prediction = predict_image(model, filepath, data_transforms, CLASS_NAMES)
    return render_template('index.html', prediction=prediction, filename=filename)

if __name__ == '__main__':
    app.run(debug=True)
