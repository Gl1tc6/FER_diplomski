import csv
from itertools import product
from typing import Optional
from torch.utils.data import DataLoader,Dataset 
from dataclasses import dataclass
import pandas as pd
from sklearn.metrics import confusion_matrix
import torch, torch.nn as nn, torch.nn.functional as F
from torch import optim
import numpy as np
import argparse

def print_progress(current_iter, max_iters, label, bar_width=50):
    if max_iters <= 0:
        fraction = "?/?"
        percentage = "?%"
        ratio = 0.0
    else:
        ratio = current_iter / max_iters
        ratio = max(0.0, min(ratio, 1.0))
        padding = len(str(max_iters)) - len(str(current_iter))
        fraction = str(current_iter) + " "*padding +"/"+ str(max_iters)
        percentage = f"{ratio * 100:.1f}%"
    
    num_full = int(ratio * bar_width)
    num_empty = bar_width - num_full
    bar = "[" + "█" * num_full + "░" * num_empty + "]"
    bar = list(bar)
    bar.insert(len(bar)//2, f" {fraction} ")
    bar = "".join(bar)
    progress_string = f"[{label}]: {bar} ({percentage})"
    
    if max_iters > 0 and current_iter >= max_iters:
        print("\r" + " "*len(progress_string) + "\r",end='', flush=True)
    else:
        print("\r" + progress_string, end='', flush=False)

# ------------------ TASK1 START ------------------ #
@dataclass
class Instance():
    def __init__(self, text, label):
        self.text = text.split(" ")
        self.label = label


class Vocab:
    """Creates a vocabulary as 2 dictionaries (stoi - StringTOIndex and itos - IndexTOString)
    """
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
    def __init__(self, path, txt_vocab:Vocab=None, lbl_vocab:Vocab=None, max_size=-1, min_freq=1):
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
            self.tv = Vocab(wrd_freq, max_size, min_freq)

        if lbl_vocab != None:   
            self.lv=lbl_vocab
        else:
            self.lv = Vocab(lbl_freq, max_size, min_freq, isLabel=True)

    def __len__(self):
        return len(self.inst)

    def __getitem__(self, idx):
        return self.tv.encode(self.inst[idx].text), self.lv.encode(self.inst[idx].label)

    def instance(self, x):
        return self.inst[x]

def pad_collate_fn(batch, pad_index=0):
    texts, labels = zip(*batch)
    lengths = torch.tensor([len(text) for text in texts])
    texts = nn.utils.rnn.pad_sequence(texts, batch_first=True, padding_value=pad_index)
    labels = torch.stack(labels)
    return texts, labels, lengths
# ------------------ TASK1 END ------------------ #
#####################################################
# ------------------ TASK2 START ------------------ #
class Embedder(nn.Module):
    def __init__(self, vocab, embedding_file = None, embedding_dim=300):
        super(Embedder, self).__init__()
        
        self.embedding = nn.Embedding(len(vocab), embedding_dim, vocab.stoi['<PAD>'])

        if embedding_file:
            with open(embedding_file, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) == embedding_dim + 1:
                        word = parts[0]
                        if word in vocab.stoi:
                            if word == '<PAD>':
                                idx = vocab.stoi[word]
                                vector = torch.zeros(embedding_dim)
                                self.embedding.weight.data[idx] = vector
                            else:
                                idx = vocab.stoi[word]
                                vector = torch.tensor([float(x) for x in parts[1:]])
                                self.embedding.weight.data[idx] = vector
            self.embedding.freeze = True

        
    def forward(self, X):
        return self.embedding(X)
    
def train(model, data, optimizer, criterion, args):
    model.train()    
    for batch_num, batch in enumerate(data):
        print_progress(batch_num, len(data)-1, "Training")
        optimizer.zero_grad()
        x, y, _ = batch
        logits = model(x).squeeze()
        loss = criterion(logits, y.float())
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        optimizer.step()

def evaluate(model, data, criterion):
    """Evaluate fun

    Args:
        model (_type_): model
        data (_type_): data
        criterion (_type_): criterion
        args (_type_): args

    Returns:
        float,float,float,torch.tensor(2,2): avg_loss, accuracy, f1, cm
    """
    model.eval()
    with torch.no_grad():
        total_loss = 0
        cm = torch.zeros(2, 2)
        
        for batch_num, batch in enumerate(data):
            print_progress(batch_num, len(data)-1, "Evaluation")
            x, y, _ = batch
            logits = model(x).squeeze(1)
            loss = criterion(logits, y.float())
            total_loss += loss.item()

            pred = torch.sigmoid(logits) > 0.5
            cm += confusion_matrix(y, pred, labels=[0, 1])           
        
        avg_loss = total_loss / len(data)
        accuracy = (cm[0, 0] + cm[1, 1]) / cm.sum()
        prec = cm[1, 1] / (cm[1, 1] + cm[0, 1])
        rec = cm[1, 1] / (cm[1, 1] + cm[1, 0])
        f1 = 2 * (prec * rec) / (prec + rec)
        
        # print(f'Validation loss: {avg_loss:.4f}')
        # print(f'Validation accuracy: {accuracy:.4f}')
        # print(f'Validation F1: {f1:.4f}')
        # print(f'Validation precision: {prec:.4f}')
        # print(f'Validation recall: {rec:.4f}')
        # print(f'Confusion matrix:\n{cm}')
        
        return avg_loss, accuracy, f1, cm

class AvgPool(nn.Module):
    def __init__(self):
        super(AvgPool, self).__init__()

    def forward(self, x):  
        x = x.permute(0, 2, 1)
        x = F.avg_pool1d(x, x.size(2))
        x = x.squeeze(2)
        return x
# ------------------ TASK2 END ------------------ #
#####################################################
# ------------------ TASK3 START ------------------ #
class RNNtype(nn.Module):
    def __init__(self, type, embedding, in_size, hidden_size, n_layers, dropout, bidirect=False, attention=None):
        super(RNNtype, self).__init__()
        self.embed = embedding
        if attention is not None:
            self.af = attention
            #self.attention = BahdanauAttention()
        else:
            self.af = False

        if type == 'rnn':
            self.rnn = nn.RNN(
                input_size=in_size,
                hidden_size=hidden_size,
                num_layers=n_layers,
                dropout=dropout if n_layers > 1 else 0,
                batch_first=True,
                bidirectional=bidirect
            )
        elif type == 'gru':
            self.rnn = nn.GRU(
                input_size=in_size,
                hidden_size=hidden_size,
                num_layers=n_layers,
                dropout=dropout if n_layers > 1 else 0,
                batch_first=True,
                bidirectional=bidirect
            )
        elif type == 'lstm':
            self.rnn = nn.LSTM(
                input_size=in_size,
                hidden_size=hidden_size,
                num_layers=n_layers,
                dropout=dropout if n_layers > 1 else 0,
                batch_first=True,
                bidirectional=bidirect
            )
        else:
            raise ValueError(f"Unknown RNN type: {type}")
        
        fc_in = hidden_size*2 if bidirect else hidden_size
        #TODO: add attention

        self.fc1=nn.Linear(fc_in, hidden_size)
        self.fc2=nn.Linear(hidden_size, 1)

    def forward(self, X):
        X = self.embed(X)
        h, _ = self.rnn(X)
        h = h[:, -1, :] if not self.af else None#else self.attention(h)

        X = self.fc1(h)
        X = F.relu(X)
        X = self.fc2(X)
        return X
# ------------------ TASK3 END ------------------ #
#####################################################
# ------------------ TASK4 START ------------------ #
def runTask4(embed, args, train_loader, val_loader, test_loader):
    h_sizes = [100, 150, 300]
    n_lys = [2, 5, 10]
    drops = [0.1, 0.5, 0.8]
    bidirect = [True, False]
    models = ["rnn", "gru", "lstm"]
    best_acc = {"rnn": (0, None), 
                "gru":(0, None),
                "lstm":(0, None)}
    best_loss= {"rnn": (-1, None), 
                "gru":(-1, None),
                "lstm":(-1, None)}
    
    combs = list(product(h_sizes, n_lys, drops, bidirect))
    selected = np.random.choice(len(combs), 3, replace=False)
    sel_comb = [combs[selected[0]], combs[selected[1]], combs[selected[2]]]
    print(sel_comb)

    for i, comb in enumerate(sel_comb):
        args.hidden_size, args.num_layers, args.dropout, bd =comb

        print(f"\nHidden size:{args.hidden_size}; # of layers: {args.num_layers}; Dropout: {args.dropout}; Bidirect.: {bd}")

        for typ in models:
            print(f"###### RNN({typ}) MODEL ######")
            model = RNNtype(typ, embed, 300, args.hidden_size, args.num_layers, args.dropout, bd)
            loss = nn.BCEWithLogitsLoss()
            opt = optim.Adam(model.parameters(), lr=args.lr)
            for epoch in range(1, args.epochs+1):
                train(model, train_loader, opt, loss, args)
                _, acc, _, _ = evaluate(model, val_loader, loss)
                print(f"Epoch {epoch}: valid acc= {acc}")

            avg_loss, acc, f1, cm = evaluate(model, test_loader, loss)
            if best_acc[typ][0] < acc:
                best_acc[typ] = (acc, comb)
                print(f"\nNew HIGHSCORE for {typ} with acc = {acc}")
            if best_loss[typ][0] > avg_loss or best_loss[typ][0] == -1:
                best_loss[typ] = (avg_loss, comb)
                print(f"\nNew HIGHSCORE for {typ} with loss = {avg_loss}")
    
    for model in models:
        acc, comb = best_acc[model]
        print(f"###### BEST for RNN({model}) MODEL ######")
        hs, ls, do, bd = comb
        print(f"Best acc: {acc}")
        print(f"Hidden size:{hs}; # of layers: {ls}; Dropout: {do}; Bidirect.: {bd}\n")
        loss, comb = best_loss[model]
        hs, ls, do, bd = comb
        print(f"Best loss: {loss}")
        print(f"Hidden size:{hs}; # of layers: {ls}; Dropout: {do}; Bidirect.: {bd}\n\n")



# ------------------ TASK4 END ------------------ #
#####################################################
# ------------------ TASK_add START ------------------ #

# ------------------ TASK_add END ------------------ #
#####################################################

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and evaluate models on SST dataset using word-vector representation GloVe")
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--epochs', type=int, default=5, help='Number of epochs')
    parser.add_argument('--Tbatch_size', type=int, default=10, help='Batch size for training')
    parser.add_argument('--VTbatch_size', type=int, default=32, help='Batch size for validation/testing')
    parser.add_argument('--clip', type=float, default=0.25, help='Gradient clipping value')
    parser.add_argument('--max_size', type=int, default=-1, help='Max vocab size')
    parser.add_argument('--min_freq', type=int, default=1, help='Min frequency for vocab')
    parser.add_argument('--seed', type=int, default=7052020, help='Random seed')
    parser.add_argument('--embedding_file', type=str, default='data/sst_glove_6b_300d.txt', help='Path to embedding file')
    parser.add_argument('--train_file', type=str, default='data/sst_train_raw.csv', help='Path to train csv')
    parser.add_argument('--val_file', type=str, default='data/sst_valid_raw.csv', help='Path to validation csv')
    parser.add_argument('--test_file', type=str, default='data/sst_test_raw.csv', help='Path to test csv')
    parser.add_argument('--base', action='store_true', help='Use baseline model')
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--test_hyper', action='store_true')
    parser.add_argument('--rnn', type=str,choices=['rnn', 'gru', 'lstm'], help='Use RNN model')
    parser.add_argument('--dropout', type=float, default=0.25, help='Gradient clipping value')
    parser.add_argument('--num_layers', type=int, default=3, help='Number of layers in RNN')
    parser.add_argument('--hidden_size', type=int, default=300, help='Number of hidden size in RNN')
    
    args = parser.parse_args()

    #np.random.seed(args.seed)
    #torch.manual_seed(args.seed)

    train_dataset = NLPDataset(args.train_file)
    val_dataset = NLPDataset(args.val_file, train_dataset.tv, train_dataset.lv)
    test_dataset = NLPDataset(args.test_file, train_dataset.tv, train_dataset.lv)

    train_loader = DataLoader(train_dataset, batch_size=args.Tbatch_size, shuffle=True, collate_fn=pad_collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=args.VTbatch_size, shuffle=False, collate_fn=pad_collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=args.VTbatch_size, shuffle=False, collate_fn=pad_collate_fn)

    if args.embedding_file:
        embed = Embedder(train_dataset.tv, args.embedding_file, embedding_dim=300)
    else:
        embed = Embedder(train_dataset.tv, embedding_dim=300)

    if args.test_hyper:
        runTask4(embed, args, train_loader, val_loader, test_loader)
        exit(0)

    model = None
    if args.base:
        print(" ###### BASELINE MODEL ###### ")
        model = nn.Sequential(
            embed,
            AvgPool(),
            nn.Linear(300, 150),
            nn.ReLU(),
            nn.Linear(150, 150),
            nn.ReLU(),
            nn.Linear(150, 1)
        )
    elif args.rnn != "":
        print(f"###### RNN({args.rnn}) MODEL ######")
        model = RNNtype(args.rnn, embed, 300, args.hidden_size, args.num_layers, args.dropout, True)
    
    if model != None:
        loss = nn.BCEWithLogitsLoss()
        opt = optim.Adam(model.parameters(), lr=args.lr)
        for epoch in range(1, args.epochs+1):
            train(model, train_loader, opt, loss, args)
            _, acc, _, _ = evaluate(model, val_loader, loss)
            print(f"Epoch {epoch}: valid acc= {acc}")

        avg_loss, acc, f1, cm = evaluate(model, test_loader, loss)
        print(f"Test acc = {acc}")
        if args.verbose:
                print(f"Avg_loss = {avg_loss}\nF1 = {f1} \nConf. matrix = \n{cm}")