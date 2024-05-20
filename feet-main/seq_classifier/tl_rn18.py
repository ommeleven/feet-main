import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from sklearn.metrics import confusion_matrix
import os
from PIL import Image
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import numpy as np

from path import remote_path

my_path = remote_path
print(my_path)

class CustomImageDataset(Dataset):
    def __init__(self, root_dir, transform=None, target_transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.target_transform = target_transform

        self.classes = ['fracture', 'edema', 'healthy']
        self.image_paths = []
        self.labels = []
        self.image_names = []

        # Load images and labels
        for label_idx, label in enumerate(self.classes):
            label_path = os.path.join(root_dir, label)
            if os.path.isdir(label_path):
                for image_name in os.listdir(label_path):
                    image_path = os.path.join(label_path, image_name)
                    self.image_paths.append(image_path)
                    self.labels.append(label_idx)  # Assigning index as label
                    self.image_names.append(image_name)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert('L')  # Convert to grayscale
        #image = image.convert('RGB')  # Convert to RGB
        image = image.resize((224, 224))  # Resize image to 224x224

        label = self.labels[idx]
        image_name = self.image_names[idx]

        if self.transform:
            image = self.transform(image)

        if self.target_transform:
            label = self.target_transform(label)

        return image, label, image_name

def preprocess(train, val):
    transformations = transforms.Compose([
        transforms.ToTensor(),
    ])

    # Define the dataset
    train_set = CustomImageDataset(train, transform=transformations)
    val_set = CustomImageDataset(val, transform=transformations)

    # Define the batch size
    batch_size = 30
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=batch_size, shuffle=True)
    
    return train_set, val_set, train_loader, val_loader
def train_healthy_unhealthy(epochs):
    model = torchvision.models.resnet18(pretrained=True)
    
    # Change the input channels of the first convolutional layer to 1
    model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)  # Changed learning rate to 0.001

    train_set, val_set, train_loader, val_loader = preprocess(train_folder, test_folder)

    train_loss_list = []
    val_loss_list = []
    train_acc_list = []
    val_acc_list = []

    confusion_matrices = []

    for epoch in range(epochs):
        train_loss = 0
        val_loss = 0
        train_accuracy = 0
        val_accuracy = 0
        y_pred_train = []
        y_true_train = []
        y_pred_val = []
        y_true_val = []
        y_name_train = []
        y_name_val = []

        model.train()
        for inputs, labels, img_names in train_loader:
            inputs, labels, img_names = inputs.to(device), labels.to(device), img_names
            optimizer.zero_grad()
            output = model.forward(inputs)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * inputs.size(0)
            
            ps = torch.exp(output)  
            top_p, top_class = ps.topk(1, dim=1)
            equals = top_class == labels.view(*top_class.shape)
            train_accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

            y_pred_train.extend(top_class.squeeze().tolist())
            y_true_train.extend(labels.tolist())
            y_name_train.extend(img_names)
        
        train_loss = train_loss / len(train_loader.dataset)
        train_accuracy = train_accuracy / len(train_loader)
        train_loss_list.append(train_loss)
        train_acc_list.append(train_accuracy)

        model.eval()
        with torch.no_grad():
            for inputs, labels, img_names in val_loader:
                inputs, labels, img_names = inputs.to(device), labels.to(device), img_names
                output = model.forward(inputs)
                valloss = criterion(output, labels)
                val_loss += valloss.item() * inputs.size(0)
                
                output = torch.exp(output)
                top_p, top_class = output.topk(1, dim=1)
                equals = top_class == labels.view(*top_class.shape)
                val_accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

                y_pred_val.extend(top_class.squeeze().tolist())
                y_true_val.extend(labels.tolist())
                y_name_val.extend(img_names)
                
        val_loss = val_loss / len(val_loader.dataset)
        val_accuracy = val_accuracy / len(val_loader)
        val_loss_list.append(val_loss)
        val_acc_list.append(val_accuracy)

        print('Train Accuracy (Healthy/Unhealthy): ', train_accuracy)
        print('Validation Accuracy (Healthy/Unhealthy): ', val_accuracy)
        print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(epoch, train_loss, val_loss))

        train_matrix = confusion_matrix(y_true_train, y_pred_train)
        print("Confusion matrix for training set (Healthy/Unhealthy):")
        print(train_matrix)

        val_matrix = confusion_matrix(y_true_val, y_pred_val)
        print("Confusion matrix for validation set (Healthy/Unhealthy):")
        print(val_matrix)
        
        confusion_matrices.append({'Epoch': epoch, 'Train_matrix': train_matrix, 'Val_matrix': val_matrix})

    plot_results(train_loss_list, val_loss_list, train_acc_list, val_acc_list, save_path='/home/odange/repo/feet_fracture_data/feet_fracture_data/graph_results/sequential/e50_lr_001_tr.png')
    save_confusion_matrices(confusion_matrices)
    
    return model



def train_fracture_edema(model, epochs):
    # Unfreeze model weights
    for param in model.parameters():
        param.requires_grad = True
    
    # Modify the final layer to classify into fracture and edema
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 2)  # 2 classes: fracture and edema

    device = 'cuda'

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    train_set, val_set, train_loader, val_loader = preprocess(train_folder, test_folder)

    train_loss_list = []
    val_loss_list = []
    train_acc_list = []
    val_acc_list = []

    confusion_matrices = []

    # Move the model to device
    model = model.to(device)

    for epoch in range(epochs):
        train_loss = 0
        val_loss = 0
        train_accuracy = 0
        val_accuracy = 0
        y_pred_train = []
        y_true_train = []
        y_pred_val = []
        y_true_val = []
        y_name_train = []
        y_name_val = []

        model.train()
        for inputs, labels, img_names in train_loader:
            inputs, labels, img_names = inputs.to(device), labels.to(device), img_names
            optimizer.zero_grad()
            output = model.forward(inputs)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * inputs.size(0)
            
            ps = torch.exp(output)  
            top_p, top_class = ps.topk(1, dim=1)
            equals = top_class == labels.view(*top_class.shape)
            train_accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

            y_pred_train.extend(top_class.squeeze().tolist())
            y_true_train.extend(labels.tolist())
            y_name_train.extend(img_names)
        
        train_loss = train_loss / len(train_loader.dataset)
        train_accuracy = train_accuracy / len(train_loader)
        train_loss_list.append(train_loss)
        train_acc_list.append(train_accuracy)

        model.eval()
        with torch.no_grad():
            for inputs, labels, img_names in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                output = model.forward(inputs)
                valloss = criterion(output, labels)
                val_loss += valloss.item() * inputs.size(0)
                
                output = torch.exp(output)
                top_p, top_class = output.topk(1, dim=1)
                equals = top_class == labels.view(*top_class.shape)
                val_accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

                y_pred_val.extend(top_class.squeeze().tolist())
                y_true_val.extend(labels.tolist())
                y_name_val.extend(img_names)
                
        val_loss = val_loss / len(val_loader.dataset)
        val_accuracy = val_accuracy / len(val_loader)
        val_loss_list.append(val_loss)
        val_acc_list.append(val_accuracy)

        print('Train Accuracy (Fracture/Edema): ', train_accuracy)
        print('Validation Accuracy (Fracture/Edema): ', val_accuracy)
        print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(epoch, train_loss, val_loss))

        train_matrix = confusion_matrix(y_true_train, y_pred_train)
        print("Confusion matrix for training set (Fracture/Edema):")
        print(train_matrix)

        val_matrix = confusion_matrix(y_true_val, y_pred_val)
        print("Confusion matrix for validation set (Fracture/Edema):")
        print(val_matrix)
        
        confusion_matrices.append({'Epoch': epoch, 'Train_matrix': train_matrix, 'Val_matrix': val_matrix})

    plot_results(train_loss_list, val_loss_list, train_acc_list, val_acc_list, save_path='/home/odange/repo/feet_fracture_data/feet_fracture_data/graph_results/sequential/e50_lr_001_tr.png')
    save_confusion_matrices(confusion_matrices)


def plot_results(train_loss_list, val_loss_list, train_acc_list, val_acc_list, save_path=None):
    epochs = range(1, len(train_loss_list) + 1)

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_loss_list, 'b', label='Training loss')
    plt.plot(epochs, val_loss_list, 'r', label='Validation loss')
    plt.title('Training and validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.ylim(0, 1)  # Set y-axis limits for loss
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_acc_list, 'b', label='Training accuracy')
    plt.plot(epochs, val_acc_list, 'r', label='Validation accuracy')
    plt.title('Training and validation accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.ylim(0, 1)  # Set y-axis limits for accuracy
    plt.legend()

    plt.tight_layout()
    
    if save_path:
        # Create the directory if it doesn't exist
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, format='png')
    else:
        plt.show()
    

def save_confusion_matrices(confusion_matrices):
    df_confusion_matrices = pd.DataFrame(confusion_matrices)
    confusion_matrices_file_path = os.path.join(directory, 'accuracy_predictions','sequential','confusion_matrices_Q8_e50_lr_001_tr.csv')
    df_confusion_matrices.to_csv(confusion_matrices_file_path, index=False)    

def main():
    model = train_healthy_unhealthy(50)
    train_fracture_edema(model, 50)

if __name__ == '__main__':
    root_dir = os.path.join(my_path)
    train_folder = os.path.join(root_dir, '3-way', 'train_st')
    test_folder = os.path.join(root_dir, '3-way', 'test_st')
    directory = os.path.join(root_dir)
    
    main()

