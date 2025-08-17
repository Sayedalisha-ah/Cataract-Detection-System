import os
import time
import copy
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import timm
from PIL import Image
import matplotlib.pyplot as plt

# Set device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Set paths for training and testing data
data_dir = 'processed_images/'  # Change to your dataset folder path

# Define data transformations
data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((224, 224)),
        # Uncomment the following for data augmentation if desired:
        # transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],  # ImageNet stats
                             std=[0.229, 0.224, 0.225])
    ]),
    'test': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
}


def predict_image(model, image_path, transform, class_names):
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0)  # add batch dimension
    image_tensor = image_tensor.to(device)
    model.eval()
    with torch.no_grad():
        output = model(image_tensor)
        _, predicted = torch.max(output, 1)
    return class_names[predicted.item()]


def plot_metrics(train_losses, train_accs, test_losses, test_accs):
    epochs = range(1, len(train_losses) + 1)
    plt.figure(figsize=(12, 5))

    # Plot Loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, label='Train Loss')
    plt.plot(epochs, test_losses, label='Test Loss')
    plt.title('Loss per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    # Plot Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accs, label='Train Accuracy')
    plt.plot(epochs, test_accs, label='Test Accuracy')
    plt.title('Accuracy per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.show()


def predict_and_plot_images(model, dataset, transform, class_names, num_images=5):
    # Randomly select num_images indices from the dataset
    indices = random.sample(range(len(dataset)), num_images)
    images = []
    predictions = []
    true_labels = []

    for idx in indices:
        img, label = dataset[idx]
        # Convert tensor to PIL image (reverse normalization for display)
        img_disp = transforms.ToPILImage()(img.cpu())
        images.append(img_disp)
        true_labels.append(class_names[label])

        # Save image temporarily to use predict_image (or alternatively predict from tensor)
        # Here we predict directly from the tensor
        img_tensor = img.unsqueeze(0).to(device)
        model.eval()
        with torch.no_grad():
            output = model(img_tensor)
            _, predicted = torch.max(output, 1)
        predictions.append(class_names[predicted.item()])

    # Plot the images with their predictions
    plt.figure(figsize=(15, 5))
    for i in range(num_images):
        plt.subplot(1, num_images, i + 1)
        plt.imshow(images[i])
        plt.title(f"True: {true_labels[i]}\nPred: {predictions[i]}")
        plt.axis('off')
    plt.tight_layout()
    plt.show()


def main():
    # Create ImageFolder datasets for training and testing
    image_datasets = {
        'train': datasets.ImageFolder(root=os.path.join(data_dir, 'train'), transform=data_transforms['train']),
        'test': datasets.ImageFolder(root=os.path.join(data_dir, 'test'), transform=data_transforms['test'])
    }

    # Create DataLoaders for training and testing
    dataloaders = {
        'train': DataLoader(image_datasets['train'], batch_size=32, shuffle=True, num_workers=4),
        'test': DataLoader(image_datasets['test'], batch_size=32, shuffle=False, num_workers=4)
    }

    # Get class names (e.g., ['cataract', 'normal'])
    class_names = image_datasets['train'].classes
    num_classes = len(class_names)
    print("Classes:", class_names)

    # Create a pretrained ViT model using timm
    model = timm.create_model('vit_base_patch16_224', pretrained=True)
    # Replace the classifier head with a new one suited to our number of classes
    model.head = nn.Linear(model.head.in_features, num_classes)
    model = model.to(device)

    # Define loss criterion and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    # Training parameters
    num_epochs = 10

    # Lists to store metrics
    train_losses, train_accs = [], []
    test_losses, test_accs = [], []

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print(f'Epoch {epoch + 1}/{num_epochs}')
        print('-' * 20)

        # To accumulate metrics per epoch
        epoch_train_loss = 0.0
        epoch_train_corrects = 0
        epoch_test_loss = 0.0
        epoch_test_corrects = 0

        # Training phase
        model.train()
        for inputs, labels in dataloaders['train']:
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            epoch_train_loss += loss.item() * inputs.size(0)
            epoch_train_corrects += torch.sum(preds == labels.data)

        train_loss = epoch_train_loss / len(image_datasets['train'])
        train_acc = epoch_train_corrects.double() / len(image_datasets['train'])
        train_losses.append(train_loss)
        train_accs.append(train_acc.item())


        # Testing phase
        model.eval()
        with torch.no_grad():
            for inputs, labels in dataloaders['test']:
                inputs = inputs.to(device)
                labels = labels.to(device)
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)

                epoch_test_loss += loss.item() * inputs.size(0)
                epoch_test_corrects += torch.sum(preds == labels.data)

        test_loss = epoch_test_loss / len(image_datasets['test'])
        test_acc = epoch_test_corrects.double() / len(image_datasets['test'])
        test_losses.append(test_loss)
        test_accs.append(test_acc.item())

        print(f'Train Loss: {train_loss:.4f}  Train Acc: {train_acc:.4f}')
        print(f'Test  Loss: {test_loss:.4f}  Test  Acc: {test_acc:.4f}')

        # Save best model based on test accuracy
        if test_acc > best_acc:
            best_acc = test_acc
            best_model_wts = copy.deepcopy(model.state_dict())
        print()

    print(f'Best Test Accuracy: {best_acc:.4f}')

    # Load best model weights
    model.load_state_dict(best_model_wts)

    # Save the trained model
    torch.save(model.state_dict(), 'vit_cataract_model.pth')
    print("Model saved as 'vit_cataract_model.pth'")

    # Plot the training metrics
    plot_metrics(train_losses, train_accs, test_losses, test_accs)

    # Predict and plot five images from the test dataset
    predict_and_plot_images(model, image_datasets['test'], data_transforms['test'], class_names, num_images=5)

    # Example usage of prediction function with a specific image path:
    sample_image_path = 'processed_images/test/cataract/image_246.png'  # Change this to an actual image path
    prediction = predict_image(model, sample_image_path, data_transforms['test'], class_names)
    print(f'Prediction for {sample_image_path}: {prediction}')


if __name__ == '__main__':
    main()
