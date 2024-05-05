#!/usr/bin/env python
# coding: utf-8


import nltk
import json
import torch
import torch.nn as nn
import numpy as np
import torch.nn as NN 
from torch.utils.data import Dataset, DataLoader
from nltk_utils import tokenize, stem, bag_of_words

with open('intents.json', 'r') as f:
    intents = json.load(f)

device =torch.device('cuda' if torch.cuda.is_available() else 'cpu')

all_words =[]
tags=[]
xy=[]
for intent in intents['intents']:
    tag= intent['tag']
    tags.append(tag)
    for pattern in intent['patterns']:
        w=tokenize(pattern)
        all_words.extend(w)
        xy.append((w, tag, ))
ignore_words = ['?', '!', '.', ',']
all_words = [stem(w) for w in all_words if w not in ignore_words]
all_words = sorted(set(all_words))
tags = sorted(set(tags))
print(tags)
print(xy)





X_train =[]
Y_train =[]
for (pattern_sentence, tag) in xy:
    bag = bag_of_words(pattern_sentence, all_words)
    X_train.append(bag)

    label = tags.index(tag)
    Y_train.append(label)

X_train =np.array(X_train)
Y_train=np.array(Y_train)





print(X_train)
print(Y_train)







class ChatDataset(Dataset):
    def __init__(self):
        self.n_samples= len(X_train)
        self.x_data=X_train
        self.y_data=Y_train
  
    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.n_samples

    
dataset= ChatDataset()
train_loader = DataLoader(dataset=dataset, batch_size=8, shuffle=True, num_workers=0)


output_size=len(tags)
input_size=len(X_train[0])
print(input_size, len(all_words))
print(output_size, tags)

class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet,self).__init__()
        self.l1=nn.Linear(input_size, hidden_size)
        self.l2=nn.Linear(hidden_size, hidden_size)
        self.l3=nn.Linear(hidden_size, num_classes)

        self.relu=nn.ReLU()

    def forward(self,x ):
        out =self.l1(x)
        out=self.relu(out)
        out=self.l2(out)
        out=self.relu(out)
        out =self.l3(out)
        out=self.relu(out)

        return out


hidden_size=6
model =NeuralNet(input_size, hidden_size, output_size).to(device)

num_epochs=700

lr=0.001
batch_size=8

criterion =nn.CrossEntropyLoss()
optimizer= torch.optim.Adam(model.parameters(), lr=lr)
for epoch in range (num_epochs):
    for (words, labels) in train_loader:
        
        words= words.to(device)
        labels= labels.to(dtype=torch.long).to(device)
        optimizer.zero_grad()

        outputs=model(words)
        loss = criterion(outputs, labels)

      
        loss.backward()
        optimizer.step()
    if (epoch +1)%100==0:
     print(f'epoch{epoch+1}/{num_epochs}, loss={loss.item():.4f}')   
    
print(f'final loss, loss={loss.item():.4f}')   


data = {
"model_state": model.state_dict(),
"input_size": input_size,
"hidden_size": hidden_size,
"output_size": output_size,
"all_words": all_words,
"tags": tags
}

FILE = "data.pth"
torch.save(data, FILE)

print(f'training complete. file saved to {FILE}')




