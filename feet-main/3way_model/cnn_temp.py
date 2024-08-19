import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.nn as nn
import torchvision
from torchvision.models import resnet18, ResNet18_Weights
from torch import Tensor
from typing import Type
import torch.optim as optim
import torchvision.models as models
import pandas as pd
from sklearn.metrics import confusion_matrix
import csv
import os
from PIL import Image
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import numpy as np
import SimpleITK as sitk
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

    def intensity_normalization(self, image):
        # Convert image to 32-bit floating-point
        image = image.astype(np.float32)
        # Intensity normalization using Nyul et al. (2000) method
        input_image = sitk.GetImageFromArray(image)
        filter_intensity = sitk.N4BiasFieldCorrectionImageFilter()
        output_image = filter_intensity.Execute(input_image)
        return sitk.GetArrayFromImage(output_image)

    def channel_wise_normalization(self, image):
        # Channel-wise normalization to have zero mean and unit variance
        mean = np.mean(image, axis=(0, 1, 2), keepdims=True)
        std = np.std(image, axis=(0, 1, 2), keepdims=True)
        return (image - mean) / std

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert('L')  # Convert to grayscale
        #image = image.resize((512, 512))  # Resize image to 512x512
        image = image.resize((224, 224))  # Resize image to 224x224

        # Apply intensity normalization
        #image = self.intensity_normalization(np.array(image))

        # Apply channel-wise normalization
        #image = self.channel_wise_normalization(image)

        label = self.labels[idx]
        image_name = self.image_names[idx]

        if self.transform:
            image = self.transform(image)

        if self.target_transform:
            label = self.target_transform(label)

        return image, label, image_name

class MyNetwork(nn.Module):
    def __init__(self): 
        self.device = torch.device('cuda')
        print(self.device)
        
        super(MyNetwork, self).__init__()
        
        self.features = nn.Sequential(
            nn.Conv2d(1, 6, 5),  # Only one input channel for grayscale images
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(6, 16, 5),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )

        #  size for the first linear layer:
        # (512 - 4) / 2 = 254 (after first pooling)
        # (254 - 4) / 2 = 125 (after second pooling)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(53*53*16, 1024),  # Update the input size here
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 3),  # Output for 2 classes
            nn.LogSoftmax(dim=1)
        )

        '''
        self.features = nn.Sequential(
            nn.Conv2d(1, 6, 5),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(6, 16, 5),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),            
        )

        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(253*253*16, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 3),  # Adjusted for 3 classes
            nn.LogSoftmax(dim=1)
        )
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(41472, 256),  # Adjusted input size from 53*53*16 to 41472
            nn.ReLU(),
            nn.Linear(256, 3),  # Adjusted for 3 classes
            nn.LogSoftmax(dim=1)
        )
        '''
        

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

class BasicBlock(nn.Module):
    def __init__(
        self, 
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        expansion: int = 1,
        downsample: nn.Module = None
    ) -> None:
        super(BasicBlock, self).__init__()
        # Multiplicative factor for the subsequent conv2d layer's output channels.
        # It is 1 for ResNet18 and ResNet34.
        self.expansion = expansion
        self.downsample = downsample
        self.conv1 = nn.Conv2d(
            in_channels, 
            out_channels, 
            kernel_size=3, 
            stride=stride, 
            padding=1,
            bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(
            out_channels, 
            out_channels*self.expansion, 
            kernel_size=3, 
            padding=1,
            bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels*self.expansion)

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return  out

class ResNet(nn.Module):
    def __init__(
        self, 
        img_channels: int,
        num_layers: int,
        block: Type[BasicBlock],
        num_classes: int  = 3
    ) -> None:
        super(ResNet, self).__init__()
        if num_layers == 18:
            
            layers = [2, 2, 2, 2]
            self.expansion = 1
        
        self.in_channels = 64
        self.conv1 = nn.Conv2d(
            in_channels=img_channels,
            out_channels=self.in_channels,
            kernel_size=7, 
            stride=2,
            padding=3,
            bias=False
        )
        self.bn1 = nn.BatchNorm2d(self.in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512*self.expansion, num_classes)

    def _make_layer(
        self, 
        block: Type[BasicBlock],
        out_channels: int,
        blocks: int,
        stride: int = 1
    ) -> nn.Sequential:
        downsample = None
        if stride != 1:
          
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.in_channels, 
                    out_channels*self.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False 
                ),
                nn.BatchNorm2d(out_channels * self.expansion),
            )
        layers = []
        layers.append(
            block(
                self.in_channels, out_channels, stride, self.expansion, downsample
            )
        )
        self.in_channels = out_channels * self.expansion

        for i in range(1, blocks):
            layers.append(block(
                self.in_channels,
                out_channels,
                expansion=self.expansion
            ))
        return nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        # The spatial dimension of the final layer's feature 
        # map should be (7, 7) for all ResNets.
        #print('Dimensions of the last convolutional feature map: ', x.shape)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

def preprocess(train, val):
    transformations = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),  # Ensure the image is single-channel
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    #transforms.Normalize([0.485], [0.229])  # Normalization for single-channel image
    #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    #transforms.Normalize(mean=[0.485], std=[0.229])
    ])
    # Define the dataset
    train_set = CustomImageDataset(train, transform=transformations)
    val_set = CustomImageDataset(val, transform=transformations)

    # Define the batch size
    batch_size = 16
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=batch_size, shuffle=True)
    
    return train_set, val_set, train_loader, val_loader

def train(epochs):
    
    #model = MyNetwork()
    #model = ResNet(img_channels=1, num_layers=18, block=BasicBlock, num_classes=3)  
    model = torchvision.models.resnet18(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, 3)
    device = 'cuda'
    model.conv1 = nn.Conv2d(1, model.conv1.out_channels, kernel_size=model.conv1.kernel_size, 
                                        stride=model.conv1.stride, padding=model.conv1.padding, bias=False)
        

    #weight_for_edema = 0.36
    #weight_for_fracture = 0.14
    #weight_for_healthy = 0.5
    #class_weights = torch.tensor([0.36, 0.14, 0.50], dtype=torch.float32).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    #optimizer = optim.SGD(model.parameters(), lr=0.003, momentum=0.9)
    
    '''
    #model = torchvision.models.resnet18(weights=ResNet18_Weights.DEFAULT)
    #model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    model = torchvision.models.resnet18(weights=ResNet18_Weights.DEFAULT)
    
    # Modify the first convolutional layer to accept single-channel (grayscale) input
    model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    
    # initialize the weights based on the existing model
    with torch.no_grad():
        model.conv1.weight = nn.Parameter(model.conv1.weight.mean(dim=1, keepdim=True))
    
    device = 'cuda'
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    
    # Freeze model weights
    for param in model.parameters():
        param.requires_grad = True
    '''
    # Modify the final layer to match the number of classes in your dataset
    #num_ftrs = model.fc.in_features
    #model.fc = nn.Linear(num_ftrs, 3)  # 3 classes: fracture, edema, healthy
    #model.fc = torch.nn.Linear(model.fc.in_features, 3)
    
    csv_directory = os.path.join(directory, 'accuracy_epoch_conf_e50.csv')

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
            inputs, labels, model = inputs.to(device), labels.to(device), model.to(device)
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

        print('Train Accuracy: ', train_accuracy)
        print('Validation Accuracy: ', val_accuracy)
        print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(epoch, train_loss, val_loss))

        train_matrix = confusion_matrix(y_true_train, y_pred_train)
        print("Confusion matrix for training set:")
        print(train_matrix)

        val_matrix = confusion_matrix(y_true_val, y_pred_val)
        print("Confusion matrix for validation set:")
        print(val_matrix)
        
        confusion_matrices.append({'Epoch': epoch, 'Train_matrix': train_matrix, 'Val_matrix': val_matrix})

    plot_results(train_loss_list, val_loss_list, train_acc_list, val_acc_list, save_path='/home/odange/repo/feet_fracture_data/graph_results/3_way/e50.png')
    save_confusion_matrices(confusion_matrices)
    #print(model.fc1(x).size())

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
    #confusion_matrices_file_path = os.path.join(directory, 'accuracy_predictions','3_way','confusion_matrices_Q8_e50.csv')
    confusion_matrices_file_path = '/home/odange/repo/feet_fracture_data/accuracy_predictions/3_way/confusion_matrices_Q8_e50.csv'
    df_confusion_matrices.to_csv(confusion_matrices_file_path, index=False)    

def main():

    #torch.cuda.empty_cache()
    train(50)

if __name__ == '__main__':
    root_dir = os.path.join(my_path)
    #train_folder = os.path.join(root_dir, '3-way', 'train_s_n')
    train_folder = '/home/odange/repo/feet_fracture_data/3_way/train'
    #test_folder = os.path.join(root_dir, '3-way', 'test_s1_n')
    test_folder = '/home/odange/repo/feet_fracture_data/3_way/val'
    directory = os.path.join(root_dir)
    
    main()
