
import matplotlib.pyplot as plt
import pandas as pd
import torch
from torchtext.legacy.data import Field, TabularDataset, BucketIterator, Iterator
import torch.nn as nn
from transformers import BertTokenizer, BertForSequenceClassification
import torch.optim as optim
from torch.utils.data import Dataset, TensorDataset
import pickle
from torch.nn.utils.rnn import pad_sequence

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
from sklearn.utils import shuffle
import numpy as np
device = 'cpu'

modelDataFolder = 'alexAndBert'
modelPath = modelDataFolder + '/model.pt'
metricsPath = modelDataFolder + '/metrics.pt'


class CustomTensorDataset(Dataset):
    def __init__(self, img, label, text, bSize, name):
        self.img = torch.tensor(img, dtype = torch.float)
        self.label = torch.tensor(label, dtype = torch.long)
        self.text = text
        print(f'len(self.text): {len(self.text)}')
        self.len = len(img)
        self.bSize = bSize
        self.name = name
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    def __getitem__(self, index):
        # print(f'Wanting index: {index}')
        start = index * self.bSize
        end = (index + 1) * self.bSize
        text = [self.tokenizer.encode(t,  return_tensors="pt")[0] for t in self.text[start: end]]
        text = pad_sequence(text, batch_first  = True)
        return self.img[start:end], text, self.label[start : end]

    def __len__(self):
        return int((self.len - 1)/ self.bSize)

def buildDataSets():
    imgData = np.load('imgData.npy')
    imgData = imgData / 255
    imgData = imgData.astype('int8')
    labelData = np.load('labelData.npy')
    # textData = np.load('textTokenData.npy')
    with open('textTokenData.npy', 'rb') as f:
        textData = pickle.load(f)
    imgData, labelData, textData = shuffle(imgData, labelData, textData)
    imgData = np.expand_dims(imgData, axis = 1)
    # print(f'imgdata.shape: {imgData.shape}, labelData.shape : {labelData.shape} textData.shape: {len(textData)}')
    
    splits = [0, .7, .8, 1]
    names = ['train', 'valid', 'test']
    dSets = []#train valid test
    for i in range(len(splits) - 1):
        start = int(splits[i] * len(imgData))
        end = int(splits[i + 1] * len(imgData))
        imgs = imgData[start: end]
        labels = labelData[start: end]
        text = textData[start: end]
        dSets.append(CustomTensorDataset(imgs, labels, text, 4, names[i]))
    return dSets


class Model(nn.Module):

    def __init__(self):
        super(Model, self).__init__()

        self.alex1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.AdaptiveAvgPool2d((6, 6)), 
        )
        self.alex2 = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, 10)
        )
        
        self.bert = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels = 10)
        self.rl = nn.LeakyReLU(0.1)
        self.fc1 = nn.Linear(20, 2)

    def forward(self, img, text):
        img = self.alex1(img)
        img = torch.flatten(img, 1)
        img = self.alex2(img)
        text = self.bert(text)
        text = text.logits
        concat = torch.cat([img, text], 1)
        concat = self.rl(concat)
        return self.fc1(concat)


def save_checkpoint(save_path, model, valid_loss):
    state_dict = {'model_state_dict': model.state_dict(),
                  'valid_loss': valid_loss}
    
    torch.save(state_dict, save_path)

def load_checkpoint(load_path):    
    state_dict = torch.load(load_path)
    return state_dict['model_state_dict']


def save_metrics(save_path, train_loss_list, valid_loss_list, global_steps_list):
    state_dict = {'train_loss_list': train_loss_list,
                  'valid_loss_list': valid_loss_list,
                  'global_steps_list': global_steps_list}
    torch.save(state_dict, save_path)


def load_metrics(load_path):
    state_dict = torch.load(load_path)
    return state_dict['train_loss_list'], state_dict['valid_loss_list'], state_dict['global_steps_list']

def train(model,
          optimizer,
          train_loader,
          valid_loader):
    num_epochs = 10
    best_valid_loss = float("Inf")
    criterion = nn.CrossEntropyLoss()

    # initialize running values
    running_loss = 0.0
    valid_running_loss = 0.0
    global_step = 0
    train_loss_list = []
    valid_loss_list = []
    global_steps_list = []

    # training loop
    model.train()
    for epoch in range(num_epochs):
        for trainIdx in range(len(train_loader)):
            img, text, label = train_loader[trainIdx]
            optimizer.zero_grad()
            output = model(img, text)
            loss = criterion(output, label)

            loss.backward()
            optimizer.step()

            # update running values
            running_loss += loss.item()
            global_step += 1

            # evaluation step
            model.eval()
            with torch.no_grad():                    

                # validation loop
                for i in range(len(valid_loader)):
                    img, text, label = valid_loader[i]
                    # print(f'Valid loader img size: {img.size()}, {label.size()}')
                    output = model(img, text)
                    loss = criterion(output, label)
                    
                    valid_running_loss += loss.item()

            # evaluation
            average_train_loss = running_loss / len(train_loader)
            average_valid_loss = valid_running_loss / len(valid_loader)
            train_loss_list.append(average_train_loss)
            valid_loss_list.append(average_valid_loss)
            global_steps_list.append(global_step)

            # resetting running values
            running_loss = 0.0                
            valid_running_loss = 0.0
            model.train()

            # print progress
            print('Epoch [{}/{}], Step [{}/{}], Train Loss: {:.4f}, Valid Loss: {:.4f}'
                    .format(epoch+1, num_epochs, global_step, num_epochs*len(train_loader),
                            average_train_loss, average_valid_loss))
            
            # checkpoint
            if best_valid_loss > average_valid_loss:
                best_valid_loss = average_valid_loss
                save_checkpoint(modelPath, model, best_valid_loss)
                save_metrics(metricsPath, train_loss_list, valid_loss_list, global_steps_list)
    
    save_metrics(metricsPath, train_loss_list, valid_loss_list, global_steps_list)

model = Model()#.to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-5)
trainSet, validSet, testSet = buildDataSets()
train(model, optimizer, trainSet, validSet)

train_loss_list, valid_loss_list, global_steps_list = load_metrics(metricsPath)
print(global_steps_list)
print(train_loss_list)
print(valid_loss_list)
plt.plot(global_steps_list, train_loss_list, label='Train')
plt.plot(global_steps_list, valid_loss_list, label='Valid')
plt.xlabel('Global Steps')
plt.ylabel('Loss')
plt.legend()
plt.savefig(f'{modelDataFolder}/loss.png') 

def evaluate(model, test_loader):
    y_pred = []
    y_true = []

    model.eval()
    with torch.no_grad():
        for i in range(len(test_loader)):
                img, text, label = test_loader[i]
                output = model(img, text)
                y_pred.extend(torch.argmax(output, 1).tolist())
                y_true.extend(label.tolist())
    
    print('Classification Report:')
    print(classification_report(y_true, y_pred, labels=[0,1], digits=4))
    
    cm = confusion_matrix(y_true, y_pred, labels=[0,1])
    ax = plt.subplot()
    sns.heatmap(cm, annot=True, ax = ax, cmap='Blues', fmt="d")

    ax.set_title('Confusion Matrix')

    ax.set_xlabel('Predicted Labels')
    ax.set_ylabel('True Labels')

    ax.xaxis.set_ticklabels(['0', '1'])
    ax.yaxis.set_ticklabels(['0', '1'])
    plt.savefig(f'{modelDataFolder}/eval.png')
    
best_model = Model()#.to(device)

data = load_checkpoint(modelPath)
best_model.load_state_dict(data)

evaluate(best_model, testSet)