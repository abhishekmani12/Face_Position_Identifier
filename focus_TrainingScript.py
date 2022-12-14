

from __future__ import print_function, division

import numpy as np
import matplotlib.pyplot as plt
import time
import os
import copy
import shutil
from tqdm import tqdm


import torch
import torchvision
from torchvision import models, transforms
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn

from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2


cudnn.benchmark=True
plt.ion()

"""**Dataset Scrapping:**"""

from google.colab import drive
drive.mount('/content/drive')

!pip install -q jmd_imagescraper
from jmd_imagescraper.core import duckduckgo_search

types ='person facing camera','person looking away'
from pathlib import Path
path = Path().cwd()/"Focus/Train"
for o in types:
  duckduckgo_search(path, o, o, max_results=1000)

os.mkdir("/content/Focus/Validation")
os.mkdir("/content/Focus/Validation/facecamera")
os.mkdir("/content/Focus/Validation/lookaway")

T_face_camera="/content/Focus/Train/facecamera/"
T_look_away="/content/Focus/Train/lookaway/"

V_face_camera="/content/Focus/Validation/facecamera/"
V_look_away="/content/Focus/Validation/lookaway/"

def mover(count, from_direc, to_direc):
  counter=count
  imgs=[]
  for f in tqdm(os.listdir(from_direc)):
    path=from_direc + f
    imgs.append(path)
    counter=counter-1
    if(counter == 0):
        break   


  for img in tqdm(imgs):
    split=img.split("/")
    newpath= to_direc + split[5]
    shutil.move(img, newpath)

mover(150, T_face_camera, V_face_camera)
mover(150, T_look_away, V_look_away)

"""**Imports:**"""

from torchvision import datasets

device="cuda" if torch.cuda.is_available() else "cpu"

"""Either one:

**Transforms:**
"""

data_transform={
    
    'Train':transforms.Compose([ #Augmentation + Resize + Normalization
        
        

        transforms.RandomResizedCrop(224), 
        transforms.RandomHorizontalFlip(),
        transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 3)), 
        transforms.RandomAdjustSharpness(sharpness_factor=3), 
        transforms.ColorJitter(brightness=(0.1,0.9), saturation=(0.5,0.9)),  
        transforms.ToTensor(),                      
        transforms.Normalize([0.485, 0.456, 0.406], 
                             [0.229, 0.224, 0.225]) 
                                                    
    ]),
    
    'Validation': transforms.Compose([ #Resize + Normalization
        
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ]),
    
}

Path="/content/Focus"

"""**Split & DataLoader:**"""

!ls /content/Focus/Train/ -a

!rm -R /content/Focus/Train/.ipynb_checkpoints

!ls /content/Focus/Train/ -a

dataset={x: datasets.ImageFolder(os.path.join(Path, x),data_transform[x]) #splitting into train and val based on folder in dataset
          for x in ['Train', 'Validation']
        }

dataloaders={x:torch.utils.data.DataLoader(dataset[x], batch_size=5,shuffle=True, num_workers=4) #defining dataloader for the model, number of dataset batches=5
              for x in ['Train', 'Validation']
            }

dataset_Size={x:len(dataset[x]) for x in ['Train','Validation']} #Determining size

label=dataset['Train'].classes #fetching labels

device=torch.device("cuda:0") #prereq for passing data onto the gpu

print(dataset_Size, label)

"""**Training func:**"""

def training(model, criterion, optimizer, scheduler, epoch_count=20):
    
    start=time.time() #keep track of initial starting time for calculating elapsed time
    
    weight_of_BestModel=copy.deepcopy(model.state_dict()) #Initialize variables to save weights and accuracy of the best model that runs through the epoches
    
    acc_of_BestModel=0.0
    
    for epoch in range(epoch_count):
        
        print(f'Epoch Number: {epoch}/{epoch_count-1}') #displaying current epoch
        print('_'*20)
        
        for phase in['Train','Validation']: # 1 epoch = Training phase + Validation phase
            
            if phase=='Train':
                model.train()  #training phase
            else:
                model.eval() #eval phase
            
            running_loss=0.0
            running_corrects=0
            
            for inputs, labels in dataloaders[phase]:
                
                #passing dataset to GPU
                
                inputs=inputs.to(device)
                labels=labels.to(device)
                
                optimizer.zero_grad() #Zeroing gradients
                
                #enable gradients for forward pass
                
                with torch.set_grad_enabled(phase == 'Train'):
                    outputs=model(inputs) #passing inputs to model
                    _, prediction=torch.max(outputs,1) #(max val of output, preds)
                    loss=criterion(outputs, labels) #cross entropy fn b/w predicted output and actual label-->loss function
                   
                #Optimize for backward pass of Train
                
                    if phase == "Train":
                        loss.backward() #calc gradients for backward pass
                        optimizer.step() #update parameters for each batch iteration
                        
                #Claculating loss
                
                running_loss=running_loss+ loss.item()*inputs.size(0)
                running_corrects=running_corrects+torch.sum(prediction==labels.data)
                
            if phase=="Train":
                scheduler.step() #change learning rate for each epoch-->lr deacy
             
            #Calc loss and acc for each epoch
            
            epoch_loss=running_loss/dataset_Size[phase]
            epoch_acc=running_corrects.double() / dataset_Size[phase]
            
            print(f'{phase} --> Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
            
            #Updating acc_of_BestModel and copying weights into weight_of_BestModel if current epoch accuracy is higher  
         
            if phase=="Validation" and epoch_acc>acc_of_BestModel:
                acc_of_BestModel=epoch_acc
                weight_of_BestModel=copy.deepcopy(model.state_dict())
        
        print()
        
    #Calculating elapsed time:
    
    elapsed_time=time.time()-start
    
    print(f'Training complete in {elapsed_time // 60:.0f}m {elapsed_time % 60:.0f}s')
    #printing highest acc
    print(f'Best val Acc: {acc_of_BestModel:4f}')
    
    #loading weight_of_BestModel
    
    model.load_state_dict(weight_of_BestModel)
    return model

"""**Pred Viz func:**

Either one:

**Reset FFC layer:**

Resnet:
"""

model_ft = models.resnet50(pretrained=True)
num_ftrs = model_ft.fc.in_features

model_ft.fc = nn.Linear(num_ftrs, 2)
model_ft = model_ft.to(device)
criterion = nn.CrossEntropyLoss()
optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

"""**Fixed Feature Extraction:**"""

pretrained_model=torchvision.models.resnet34(pretrained=True) #loading pretrained model-resnet34

for param in pretrained_model.parameters():
    
    param.requires_grad=False #freezing network params except for final layer where gradients are not calculated for backward pass

#Modifying the last fc layer

num_of_ftrs=pretrained_model.fc.in_features # calculating number of inputs for the layers

pretrained_model.fc=nn.Linear(num_of_ftrs, 2)

#passing to GPU
pretrained_model=pretrained_model.to(device)

#defining loss fn:

criterion=nn.CrossEntropyLoss()

optimizer_new=optim.SGD(pretrained_model.fc.parameters(), lr=0.001, momentum=0.9) #optimizer for last fc layer only --> SGD

#LR decay by 0.1 factor for every 7 epoches

exp_lr_scheduler= lr_scheduler.StepLR(optimizer_new, step_size=7, gamma=0.1)

"""**TRAIN:**"""

pretrained_model=training(model_ft, criterion, optimizer_ft, exp_lr_scheduler, epoch_count=20)

"""**SAVE:**"""

torch.save(pretrained_model,'model.pth')
torch.save(pretrained_model.state_dict(),'model_weights.pth')