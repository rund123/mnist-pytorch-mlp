"""
mnist_mlp.py
-------------
MNIST Handwritten Digit Classification using PyTorch.

This script implements a fully connected neural network (MLP) to classify 
handwritten digits from the MNIST dataset. It includes:
- Dataset loading and preprocessing
- Model definition
- Training and validation loops
- Test evaluation
- Model saving
- Plotting training and validation loss
"""

import torch
import torch.nn as nn
import torch.nn.functional as F 
import torch.optim as optim 
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import os


# Initilizing the Hyperparameters globaly for testing purposes 

BATCH_SIZE = 250
LEARNING_RATE = 0.001
EPOCHS = 25
HIDDEN1 = 512
HIDDEN2 = 265
HIDDEN3 = 128
NUM_CLASSES = 10
MODEL_SAVE_PATH = "saved_model/mnist_mlp.pth"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

##____Data Preperation____## 

# Define transforms
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize (0.5,0.5)])

# Create the training and test sets.
training_data = datasets.MNIST(root = "data", train=True, download=True, transform=transform)
test_data = datasets.MNIST(root = "data", train=False, download=True, transform=transform)

# Create the training and test dataloaders
train_loader = DataLoader (training_data, batch_size=BATCH_SIZE, shuffle = True)
test_loader = DataLoader (test_data, batch_size=BATCH_SIZE)


##____Model Defenition____##

class Net(nn.Module):

    """Fully Connected MLP for MNIST"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.activation = F.relu
        self.input_layer = nn.Linear(28*28, HIDDEN1)
        self.hidden_layer1 = nn.Linear (HIDDEN1, HIDDEN2)
        self.hidden_layer2 = nn.Linear (HIDDEN2, HIDDEN3)
        self.output_layer = nn.Linear (HIDDEN3, NUM_CLASSES)

    def forward(self,x):
        x = torch.flatten(x,1)
        x = self.activation(self.input_layer(x))
        x = self.activation(self.hidden_layer1(x))
        x = self.activation(self.hidden_layer2(x))
        x = self.output_layer(x)
        return x

net = Net()
net.to(device)


##____Training Function____##

def train (model, train_loader, criterion, optimizer):

    # Enable dropout and batch normalization updates
    net.train()
    
    # Initilize metrics for each epoch call(total loss, correct prediction, total training sample)
    train_loss = 0.0
    corr_pred = 0.0
    total_train_sample = 0

    # loop over patches (TRAINING) 
    for images, labels in (train_loader):
    
        # Move data to GPU if available
        images, labels = images.to(device), labels.to(device)

        # ( forward Pass )
        # Data flow through the network
        outputs = model(images)

        # Calculate the loss
        loss = criterion(outputs, labels)

        # ( Backpropogation )
        
        # Zeroing the gradinebt before each backward pass since bytorch calculate it by default 
        optimizer.zero_grad()
        
        # calculate the gradinat with respect to model parameters (bakward)
        loss.backward()

        # Update weights 
        optimizer.step()

        # Calculate accurecy by returning the max value and its index
        _, label_pred = torch.max(outputs,1)

        # Count how many prediction is correct and store it in correct prediction var
        corr_pred += (label_pred == labels).sum().item()
        
        # Accumalte loss
        train_loss += loss.item()* labels.size(0)
       
        # Count total training samples 
        total_train_sample += labels.size(0)

    training_accuracy = 100 * corr_pred / total_train_sample
    training_loss = train_loss / total_train_sample

    return training_accuracy, training_loss

# Validation Function 

def validation (model, train_loader, criterion, optimizer):

    # Enable dropout and batch normalization updates
    net.eval()
    
    # Initilize metrics for each epoch (total loss val, correct prediction val, total val sample)
    val_loss = 0.0
    val_corr_pred = 0.0
    total_val_sample = 0
    
    # loop over Validation Data
    for images, labels in (test_loader):
    
        # Move data to GPU if available
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()

        # ( forward Pass )
        # Data flow through the network
        outputs = model (images)
        # Calculate the loss
        loss = criterion(outputs,labels)

        # Calculate accurecy by returning the max value and its index
        _, label_pred = torch.max(outputs,1)

        # Count how many prediction is correct and store it in correct prediction var
        val_corr_pred += (label_pred == labels).sum().item()

        # Accumalte loss
        val_loss += loss.item()* labels.size(0)
        
        # Count total validation samples 
        total_val_sample += labels.size(0)

    validation_accuracy = 100 * val_corr_pred / total_val_sample
    validation_loss = val_loss / total_val_sample

    return validation_accuracy, validation_loss

##____Plotting Function____## 

def plot (train_loss_history, val_loss_history):

    plt.plot(train_loss_history, label="Training Loss")
    plt.plot(val_loss_history, label="Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()

##____Save Model Function____##

def save_model ():

    os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
    torch.save(net.state_dict(), MODEL_SAVE_PATH)
    print(f"Model saved to {MODEL_SAVE_PATH}")

# Training Loop Main()

def main ():

    net = Net().to(device)
    optimizer = optim.SGD(net.parameters(), momentum = 0.9, lr = LEARNING_RATE)
    criterion = nn.CrossEntropyLoss() 

    training_loss_history = list()
    val_loss_history = list()

    for epoch in range (EPOCHS):

        train_acc, train_loss = train (net, train_loader, criterion, optimizer)
        val_acc, val_loss = validation (net, train_loader, criterion, optimizer)

        training_loss_history.append(train_loss)
        val_loss_history.append(val_loss)

        print (f'Epoch {epoch+1} Training Data Accuracy Rate is {train_acc:.2f}% , Training Data Loss Margin is {train_loss:.5f} ')
        print (f'Epoch {epoch+1} Validation Data Accuracy Rate is {val_acc:.2f}% , Validation Data Loss Margin is {val_loss:.5f} ')

    plot(training_loss_history, val_loss_history)



if __name__=="__main__":
    main()
