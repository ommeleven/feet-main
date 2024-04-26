import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from sklearn.metrics import confusion_matrix
import csv
import os
from PIL import Image
from torch.utils.data import Dataset

from path import my_path

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
        bw_image = Image.open(image_path)
        image = bw_image.convert("RGB")
        label = self.labels[idx]
        image_name = self.image_names[idx]

        if self.transform:
            image = self.transform(image)

        if self.target_transform:
            label = self.target_transform(label)

        return image, label, image_name
        '''
        image_path = self.image_paths[idx]
        bw_image = Image.open(image_path).convert("L")  # Convert to grayscale
        image = bw_image.resize((512, 512))  # Resize to 512x512
        image = transforms.ToTensor()(image)  # Convert to tensor
        to_pil = transforms.ToPILImage()
        image = to_pil(image)

        label = self.labels[idx]
        image_name = self.image_names[idx]

        if self.transform:
            image = self.transform(image)

        if self.target_transform:
            label = self.target_transform(label)

        return image, label, image_name
        '''

class MyNetwork(nn.Module):
    def __init__(self): 
        super(MyNetwork, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 6, 5),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(6, 16, 5),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),            
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(53*53*16, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 3),  # Adjusted for 3 classes
            nn.LogSoftmax(dim=1)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x
    

def preprocess(train, val):
    '''
    transformations = transforms.Compose([
    transforms.Resize((512, 512)), # Adjusted from 224x224 to 512x512
    transforms.ToTensor()])
    '''
    transformations = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


    # Define the dataset
    train_set = CustomImageDataset(train, transform=transformations)
    val_set = CustomImageDataset(val, transform=transformations)
    #print("train_set: ", len(train_set))
    # Define the batch size
    batch_size = 30
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=batch_size, shuffle=True)
    
    return train_set, val_set, train_loader, val_loader

def train(epochs):
    model = MyNetwork()
    device = "cpu"

    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters())
    # Adjusted directories accordingly
    csv_directory = os.path.join(directory, 'accuracy_epoch_conf.csv')

    train_set, val_set, train_loader, val_loader = preprocess(train_folder, test_folder)

    train_loss = 0
    val_loss = 0
    train_accuracy = 0
    val_accuracy = 0
    y_pred_train = []
    y_true_train = []
    y_pred_val = []
    y_true_val = []
    confusion_matrices = []

    with open(csv_directory, 'w', newline='') as f:
        fieldNames = ['Epoch', 'Train_accuracy', 'Train_loss', 'Validation_accuracy', 'Validation_loss']
        myWriter = csv.DictWriter(f, fieldnames=fieldNames)
        myWriter.writeheader()

        for epoch in range(epochs):
            train_loss = 0
            val_loss = 0
            train_accuracy = 0
            val_accuracy = 0
            y_pred_train = []
            y_true_train = []
            y_name_train = []
            y_pred_val = []
            y_true_val = []
            y_name_val = []

            model.train()
            for inputs, labels, img_names in train_loader:
                inputs, labels = inputs.to(device), labels.to(device)
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
                    
            train_loss = train_loss / len(train_loader.dataset)
            val_loss = val_loss / len(val_loader.dataset)

            print('Train Accuracy: ', train_accuracy / len(train_loader))
            print('Validation Accuracy: ', val_accuracy / len(val_loader))
            print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(epoch, train_loss, val_loss))

            train_matrix = confusion_matrix(y_true_train, y_pred_train)
            print("Confusion matrix for training set:")
            print(train_matrix)

            val_matrix = confusion_matrix(y_true_val, y_pred_val)
            print("Confusion matrix for validation set:")
            print(val_matrix)

            myWriter.writerow({'Epoch': epoch + 1, 'Train_accuracy': train_accuracy / len(train_loader), \
                                'Train_loss': train_loss, 'Validation_accuracy': val_accuracy / len(val_loader), \
                                'Validation_loss': val_loss})
            
            confusion_matrices.append({'Epoch': epoch, 'Train_matrix': train_matrix, 'Val_matrix': val_matrix})

    df_confusion_matrices = pd.DataFrame(confusion_matrices)
    confusion_matrices_file_path = os.path.join(directory, 'accuracy_predictions', 'confusion_matrices_Q8.csv')
    df_confusion_matrices.to_csv(confusion_matrices_file_path, index=False)    

def main():
    train(5)

if __name__ == '__main__':
    root_dir_864 = os.path.join(my_path, '864')
    #train_folder = os.path.join(root_dir_864, 'train_undersampling')  
    train_folder = '/Users/HP/src/feet_fracture_data/864' + '/train_oversampling'
    test_folder = os.path.join(root_dir_864, 'test')
    directory = os.path.join(root_dir_864, 'accuracy_predictions')
    
    main()
