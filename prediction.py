import os
import torch
import torch.nn as nn
from torchvision import transforms
import timm
from PIL import Image

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


if __name__ == '__main__':
    # Specify the path to the saved model weights
    model_path = 'vit_cataract_model.pth'
    # Adjust class names to match your dataset
    class_names = ['cataract', 'normal']

    # Load the model
    model = load_model(model_path, class_names)
    print("Model loaded successfully.")

    # Define the image path to predict
    image_path = 'processed_images/test/normal/image_247.png'

    # Perform prediction
    prediction = predict_image(model, image_path, data_transforms, class_names)
    print(f'Prediction for {image_path}: {prediction}')
