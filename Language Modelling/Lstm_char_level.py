import re
import string
import numpy as np

import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.optim as optim
import torch.utils.data as data

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


path = 'pg1342.txt'  # text file path
seq_length = 100

def open_file(path):
    raw_text = open(path,'r',encoding='utf-8-sig').read()
    raw_text = raw_text.lower()
    return raw_text

def preprocess(raw_text):
    text = re.sub(r'[^A-Za-z0-9 ]+',' ',raw_text)
    text = re.sub('\s+',' ',text)
    text = re.sub(r'\n','',text)
    text = re.sub(r',.;','',text)
    text = re.sub(r'\ufeff','',text)
    text = re.sub(r'\d+','',text) 
    

    bad_chars = ["●","•","|",'™','”','“']         
    for char in bad_chars:
        text = text.replace(char,"")
    return text


def word_to_index(text):
    chars = sorted(list(set(text)))
    char_to_int = dict((c,i) for i, c in enumerate(chars))

    key_list = list(char_to_int.keys())   # Character
    val_list = list(char_to_int.values()) # Numerical Index
    return chars, char_to_int

text = preprocess(open_file(path))
chars, char_to_int = word_to_index(text)


def generate_io(text):
    char_to_int = dict((c,i) for i, c in enumerate(chars))
    X_train, y_train = [],[]

    for i in range(0, len(text) - seq_length, 1):
        seq_in = text[i:i + seq_length]
        seq_out = text[i + seq_length]
        X_train.append([char_to_int[char] for char in seq_in])
        y_train.append(char_to_int[seq_out])
        
    X = torch.tensor(X_train, dtype = torch.float32).reshape(len(X_train),seq_length,1)
    X = (X / float(len(chars))).to(device)
    y = torch.tensor(y_train,dtype = torch.float32) / float(len(chars))
    y = y.to(device)
    return X,y

class CharModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.chars = chars
        self.lstm = nn.LSTM(input_size=1, hidden_size=256, num_layers=1, batch_first=True)
        self.dropout = nn.Dropout(0.2)
        self.linear = nn.Linear(256, len(chars))
    def forward(self, x):
        x, _ = self.lstm(x)
        # take only the last output
        x = x[:, -1, :]
        # produce output
        x = self.linear(self.dropout(x))
        return x

def training(text,X,y):
    chars = sorted(list(set(text)))
    char_to_int = dict((c,i) for i, c in enumerate(chars))
    n_epochs = 5
    batch_size = 512
    model = CharModel()
    model = model.to(device)

    optimizer = optim.Adam(model.parameters())
    loss_fn = nn.CrossEntropyLoss(reduction="sum")
    loader = data.DataLoader(data.TensorDataset(X, y), shuffle=True, batch_size=batch_size)

    best_model = None
    best_loss = np.inf
    loss_store = []
    for epoch in range(n_epochs):
        model.train()
        for X_batch, y_batch in loader:
            y_pred = model((X_batch.float().to(device)))
            loss = loss_fn(y_pred, y_batch.long())

            optimizer.zero_grad()
            loss.backward()

            optimizer.step()
        # Validation
        model.eval()
        loss = 0

        with torch.no_grad():
            for X_batch, y_batch in loader:
                y_pred = model((X_batch.float().to(device)))
                loss += loss_fn(y_pred, y_batch.long())
            loss_store.append(loss)
            if loss < best_loss:
                best_loss = loss
                best_model = model.state_dict()
            print(f"Epoch {epoch} : Cross-entropy: {loss}")
    model_save = torch.save([best_model, char_to_int], "single-char.pth")
    return model_save


def main():
    X, y = generate_io(text)
    model_save = training(text,X,y)

if __name__ == '__main__':
    main()
  
