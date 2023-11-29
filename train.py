import json
from nltk_utils import tokenize, stem, bag_of_words
import numpy as np
import random
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from model import LLM

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with open('LLM.json', 'r') as f:
    intents = json.load(f)

all_words = []
tags = []
xy = []

for intent in intents['intents']:
    tag = intent['tag']
    tags.append(tag)
    for pattern in intent['patterns']:
        w = tokenize(pattern)
        all_words.extend(w)
        xy.append((w, tag))

ignore = ['?', ',', '.', '!']
all_words = [stem(w) for w in all_words if w not in ignore]
all_words = sorted(set(all_words)) # remove repetitive words
tags = sorted(set(tags)) # Leave only unique labels

X_train = []
y_train = []
for (pattern_sentence, tag) in xy:
    bag = bag_of_words(pattern_sentence, all_words)
    X_train.append(bag)

    label = tags.index(tag)
    y_train.append(label)

X_train = np.array(X_train)
y_train = np.array(y_train)

class ChatBot(Dataset):
    def __init__(self):
        self.n_samples = len(X_train)
        self.x_data = X_train
        self.y_data = y_train

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]
    def __len__(self):
        return self.n_samples

BATCH_SIZE = 1
INPUT_SIZE = len(X_train[0])
HIDDEN_UNITS = 10
OUTPUT_SIZE = len(tags)
DATA = ChatBot()

train_loader = DataLoader(dataset=DATA, 
                          batch_size=BATCH_SIZE, 
                          shuffle=True,
                          num_workers=0)

model = LLM(INPUT_SIZE,
            HIDDEN_UNITS,
            OUTPUT_SIZE).to(device)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(params= model.parameters(),
                             lr = 0.001)

epochs = 1000
for epoch in range(epochs):
    for (words, labels) in train_loader:
        words = words.to(device)
        labels = labels.to(dtype=torch.long).to(device)

        labels_pred = model(words)

        loss = loss_fn(labels_pred, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    if epoch % 100 == 0:
        print(f"epoch:{epoch} | loss: {loss.item():.5f}")

results = {"model state": model.state_dict,
           "input size": INPUT_SIZE,
           "hidden units": HIDDEN_UNITS,
           "output size": OUTPUT_SIZE,
           "all words": all_words,
           "tags": tags}

FILE = "results.pth"
torch.save(results, FILE)
print(f"Training completed")