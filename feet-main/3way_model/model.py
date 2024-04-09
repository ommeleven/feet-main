import os
from time import time 
from tqdm import tqdm 
import numpy
import torch
from torch.nn import Linear, CrossEntropyLoss 
from torch.optim import Adam 
from torch.utils.data import DataLoader
import torchvision
from torchvision.datasets import ImageFolder 
from torchvision.models import resnet18
from torchvision.transforms import transforms
from path import my_path

#device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#transformer
tfm = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])


#dataset
TRAIN_DIR = my_path + '/512/train'
TEST_DIR = my_path + '/512/test'

train_ds = ImageFolder(root=TRAIN_DIR, transform=tfm)
test_ds = ImageFolder(root=TEST_DIR, transform=tfm) 
print('train_ds', train_ds.class_to_idx)
#train_ds.class_to_idx = {'fracture': 0, 'edema': 1, 'healthy': 2}
#test_ds.class_to_idx = {'fracture': 0, 'edema': 1, 'healthy': 2}

#dataloader
train_loader = DataLoader(dataset=train_ds, batch_size=32, shuffle=True)
test_loader = DataLoader(dataset=test_ds, batch_size=32, shuffle=True)

#model
model = resnet18(pretrained=True)

#replacing the last layer
model.fc = Linear(in_features=512, out_features=3)
model.to(device)

#loss and optimizer
optimizer = Adam(model.parameters(), lr=3e-4, weight_decay=0.0001)
loss_fn = CrossEntropyLoss()

#training   
for epoch in range(3):
    start = time()
    tr_acc = 0
    test_acc = 0

    #train
    model.train()

    with tqdm(train_loader, unit="batch") as tepoch:
        for xtrain, ytrain in tepoch:
            optimizer.zero_grad()
            xtrain = xtrain.to(device)
            train_prob = model(xtrain)
            train_prob = train_prob.cpu()

            loss = loss_fn(train_prob, ytrain)
            loss.backward()
            optimizer.step()
            
            train_pred = torch.max(train_prob, 1).indices
            tr_acc += int(torch.sum(train_pred == ytrain))
           
        ep_tr_acc = tr_acc / len(train_ds)
    
    model.eval()
    with torch.no_grad():
        for xtest, ytest in test_loader:
            xtest = xtest.to(device)
            test_prob = model(xtest)
            test_prob = test_prob.cpu()
            
            test_pred = torch.max(test_prob, 1).indices
            test_acc += int(torch.sum(test_pred == ytest))

        ep_test_acc = test_acc / len(test_ds)
    
    end = time()
    duration = (end - start)/ 60
    
    print(f'Epoch {epoch+1}/{3}, Train Accuracy: {ep_tr_acc:.4f}, Test Accuracy: {ep_test_acc:.4f}, Duration: {duration:.2f} minutes')

