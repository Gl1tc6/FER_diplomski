import csv
from torch.utils.data import DataLoader,Dataset 
from dataclasses import dataclass
import pandas as pd
from sklearn.metrics import confusion_matrix
import torch, torch.nn as nn, torch.nn.functional as F
from torch import optim
from tqdm import tqdm
import numpy as np

# ------------------ TASK1 START ------------------ #
@dataclass
class Instance():
    def __init__(self, text, label):
        self.text = text.split(" ")
        self.label = label


class Vocab:
    def __init__(self, freq, max_size=-1, min_freq=0, isLabel=False):
        self.stoi = {'<PAD>':0, '<UNK>':1} if not isLabel else {}
        self.itos = {0:'<PAD>', 1:'<UNK>'}
         
        freq = sorted(freq.items(), key=lambda x: x[1], reverse=True)
        
        idx = 2 if not isLabel else 0
        
        if max_size == -1:
            for wrd, wc in freq:
                if wc >= min_freq:
                    self.stoi[wrd] = idx
                    self.itos[idx] = wrd
                    idx += 1
        else:
            for i in range(max_size):
                if i >= len(freq):
                    break
                wrd, wc = freq[i]
                if wc >= min_freq:
                    self.stoi[wrd] = idx
                    self.itos[idx] = wrd
                    idx += 1
                    
    def encode(self, text):
        if isinstance(text, str):
            return torch.tensor(self.stoi.get(text, 1))
        return torch.tensor([self.stoi.get(word,1) for word in text])
    
    def __len__(self):
        return len(self.itos)
    
    def get_freq(dataset):
        wrd_freq = {}
        lbl_freq = {}
        for instance in dataset:
            for wrd in instance.text:
                wrd_freq[wrd] = wrd_freq.get(wrd, 0) + 1
            lbl_freq[instance.label] = lbl_freq.get(instance.label, 0) + 1
        return wrd_freq, lbl_freq
        
        
class NLPDataset (torch.utils.data.Dataset):
    def __init__(self, path, txt_vocab:Vocab=None, lbl_vocab:Vocab=None):
        self.inst = []
        with open(path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            for line in reader:
                if len(line) >= 2:
                    text_part = line[0].strip()
                    label_part = line[1].strip()
                    self.inst.append(Instance(text=text_part, label=label_part))
        
        wrd_freq, lbl_freq = Vocab.get_freq(self.inst)
        
        if txt_vocab != None:
            self.tv=txt_vocab
        else:
            self.tv = Vocab(wrd_freq)
        
        if lbl_vocab != None:   
            self.lv=lbl_vocab
        else:
            self.lv = Vocab(lbl_freq, isLabel=True)
    
    def __len__(self):
        return len(self.inst)
    
    def __getitem__(self, idx):
        return self.tv.encode(self.inst[idx].text), self.lv.encode(self.inst[idx].label)
    
    def instance(self, x):
        return self.inst[x]
    
def load_embed(vocab:Vocab, file:str=None, embedding_dim=300):
    size = len(vocab)
    embedding_matrix = np.random.normal(0, 1, (size, embedding_dim))
    
    embedding_matrix[0] = np.zeros(embedding_dim)
    
    if file:
        with open(file, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == embedding_dim + 1:
                    word = parts[0]
                    if word in vocab.stoi:
                        idx = vocab.stoi[word]
                        vector = np.array([float(x) for x in parts[1:]])
                        embedding_matrix[idx] = vector
    return torch.FloatTensor(embedding_matrix)

def pad_collate_fn(batch, pad_index=0):
    texts, labels = zip(*batch)
    lengths = torch.tensor([len(text) for text in texts])
    texts = nn.utils.rnn.pad_sequence(texts, batch_first=True, padding_value=pad_index)
    labels = torch.stack(labels)
    return texts, labels, lengths

# ------------------ TASK1 END ------------------ #

# ------------------ TASK2 START ------------------ #
class BaselineModel(nn.Module):
    pass

# ------------------ TASK2 END ------------------ #


if __name__ == "__main__":
    # ---- CONFIGS ---- #
    lr=1e-3
    vocab_size = -1
    min_frequency = 0
    # ---- SGIFNOC ---- #
    
    # train_dataset = NLPDataset('sst_train_raw.csv')
    # val_dataset = NLPDataset('sst_valid_raw.csv', train_dataset.tv, train_dataset.lv)
    # test_dataset = NLPDataset('sst_test_raw.csv', train_dataset.tv, train_dataset.lv)

    # train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=pad_collate_fn)
    # val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, collate_fn=pad_collate_fn)
    # test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, collate_fn=pad_collate_fn)
    # batch_size = 2 # Only for demonstrative purposes
    # shuffle = False # Only for demonstrative purposes
    # train_dataset = NLPDataset('sst_train_raw.csv')
    # train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size, 
    #                             shuffle=shuffle, collate_fn=pad_collate_fn)
    # texts, labels, lengths = next(iter(train_dataloader))
    # print(f"Texts: {texts}")
    # print(f"Labels: {labels}")
    # print(f"Lengths: {lengths}")