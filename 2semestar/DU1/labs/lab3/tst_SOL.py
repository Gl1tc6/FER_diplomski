import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import numpy as np
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional
import csv
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Task 1: Data Loading Implementation

@dataclass
class Instance:
    """Simple wrapper around data instances"""
    text: List[str]
    label: str

class Vocab:
    """Vocabulary class for string-to-index conversion"""
    
    def __init__(self, frequencies: Dict[str, int], max_size: int = -1, min_freq: int = 1, 
                 is_label: bool = False):
        self.max_size = max_size
        self.min_freq = min_freq
        self.is_label = is_label
        
        # Special symbols only for text vocabulary
        if not is_label:
            self.itos = ['<PAD>', '<UNK>']
            self.stoi = {'<PAD>': 0, '<UNK>': 1}
        else:
            self.itos = []
            self.stoi = {}
        
        # Sort by frequency (descending)
        sorted_words = sorted(frequencies.items(), key=lambda x: x[1], reverse=True)
        
        for word, freq in sorted_words:
            if freq >= min_freq:
                if max_size == -1 or len(self.itos) < max_size:
                    if word not in self.stoi:
                        self.stoi[word] = len(self.itos)
                        self.itos.append(word)
                else:
                    break
    
    def encode(self, tokens):
        """Convert tokens to indices"""
        if isinstance(tokens, str):
            # Single token
            if self.is_label:
                return torch.tensor(self.stoi.get(tokens, 0))
            else:
                return torch.tensor(self.stoi.get(tokens, 1))  # UNK index for text
        else:
            # List of tokens
            indices = []
            for token in tokens:
                if self.is_label:
                    indices.append(self.stoi.get(token, 0))
                else:
                    indices.append(self.stoi.get(token, 1))  # UNK index for text
            return torch.tensor(indices)
    
    def __len__(self):
        return len(self.itos)

class NLPDataset(Dataset):
    """Dataset class for NLP data"""
    
    def __init__(self, instances: List[Instance], text_vocab: Vocab, label_vocab: Vocab):
        self.instances = instances
        self.text_vocab = text_vocab
        self.label_vocab = label_vocab
    
    @classmethod
    def from_file(cls, filepath: str, text_vocab: Optional[Vocab] = None, 
                  label_vocab: Optional[Vocab] = None):
        """Load dataset from CSV file"""
        instances = []
        
        with open(filepath, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            for line in reader:
                if len(line) >= 2:
                    # Split text and label (format: "word1 word2 ..., label")
                    text_part = line[0].strip()
                    label_part = line[1].strip()
                    
                    text_tokens = text_part.split()
                    instances.append(Instance(text=text_tokens, label=label_part))
        
        # Build vocabularies if not provided (training data)
        if text_vocab is None:
            text_frequencies = Counter()
            label_frequencies = Counter()
            
            for instance in instances:
                text_frequencies.update(instance.text)
                label_frequencies[instance.label] += 1
            
            text_vocab = Vocab(text_frequencies, max_size=-1, min_freq=1, is_label=False)
            label_vocab = Vocab(label_frequencies, is_label=True)
        
        return cls(instances, text_vocab, label_vocab)
    
    def __len__(self):
        return len(self.instances)
    
    def __getitem__(self, idx):
        instance = self.instances[idx]
        numericalized_text = self.text_vocab.encode(instance.text)
        numericalized_label = self.label_vocab.encode(instance.label)
        return numericalized_text, numericalized_label

def load_embeddings(vocab: Vocab, embedding_file: str = None, embedding_dim: int = 300):
    """Load or initialize embedding matrix"""
    vocab_size = len(vocab)
    embedding_matrix = np.random.normal(0, 1, (vocab_size, embedding_dim))
    
    # Set padding vector to zeros
    embedding_matrix[0] = np.zeros(embedding_dim)
    
    if embedding_file:
        print(f"Loading embeddings from {embedding_file}")
        with open(embedding_file, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split(',')
                if len(parts) == embedding_dim + 1:
                    word = parts[0]
                    if word in vocab.stoi:
                        idx = vocab.stoi[word]
                        vector = np.array([float(x) for x in parts[1:]])
                        embedding_matrix[idx] = vector
    
    return torch.FloatTensor(embedding_matrix)

def pad_collate_fn(batch, pad_index=0):
    """Collate function for padding sequences"""
    texts, labels = zip(*batch)
    
    # Get lengths before padding
    lengths = torch.tensor([len(text) for text in texts])
    
    # Pad sequences
    texts = pad_sequence(texts, batch_first=True, padding_value=pad_index)
    labels = torch.stack(labels)
    
    return texts, labels, lengths

# Task 2: Baseline Model Implementation

class BaselineModel(nn.Module):
    """Baseline model with mean pooling"""
    
    def __init__(self, vocab_size: int, embedding_dim: int = 300, 
                 hidden_dim: int = 150, num_classes: int = 1,
                 pretrained_embeddings: Optional[torch.Tensor] = None,
                 freeze_embeddings: bool = True):
        super().__init__()
        
        if pretrained_embeddings is not None:
            self.embedding = nn.Embedding.from_pretrained(
                pretrained_embeddings, 
                freeze=freeze_embeddings,
                padding_idx=0
            )
        else:
            self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        
        self.fc1 = nn.Linear(embedding_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, num_classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.1)
    
    def forward(self, x, lengths):
        # x: (batch_size, seq_len)
        # lengths: (batch_size,)
        
        embedded = self.embedding(x)  # (batch_size, seq_len, embedding_dim)
        
        # Mean pooling - average over sequence length, ignoring padding
        mask = (x != 0).float().unsqueeze(-1)  # (batch_size, seq_len, 1)
        masked_embeddings = embedded * mask
        pooled = masked_embeddings.sum(dim=1) / lengths.float().unsqueeze(-1)
        
        # Feed forward layers
        x = self.relu(self.fc1(pooled))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        
        return x

# Task 3: RNN Model Implementation

class RNNModel(nn.Module):
    """RNN-based model for sentiment classification"""
    
    def __init__(self, vocab_size: int, embedding_dim: int = 300,
                 hidden_dim: int = 150, num_layers: int = 2,
                 num_classes: int = 1, rnn_type: str = 'GRU',
                 dropout: float = 0.1, bidirectional: bool = False,
                 pretrained_embeddings: Optional[torch.Tensor] = None,
                 freeze_embeddings: bool = True,
                 use_attention: bool = False):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.use_attention = use_attention
        self.rnn_type = rnn_type
        
        # Embedding layer
        if pretrained_embeddings is not None:
            self.embedding = nn.Embedding.from_pretrained(
                pretrained_embeddings, 
                freeze=freeze_embeddings,
                padding_idx=0
            )
        else:
            self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        
        # RNN layer
        if rnn_type == 'LSTM':
            self.rnn = nn.LSTM(embedding_dim, hidden_dim, num_layers,
                              dropout=dropout if num_layers > 1 else 0,
                              bidirectional=bidirectional, batch_first=True)
        elif rnn_type == 'GRU':
            self.rnn = nn.GRU(embedding_dim, hidden_dim, num_layers,
                             dropout=dropout if num_layers > 1 else 0,
                             bidirectional=bidirectional, batch_first=True)
        else:  # Vanilla RNN
            self.rnn = nn.RNN(embedding_dim, hidden_dim, num_layers,
                             dropout=dropout if num_layers > 1 else 0,
                             bidirectional=bidirectional, batch_first=True)
        
        # Attention mechanism (Bonus task)
        if use_attention:
            attention_dim = hidden_dim // 2
            rnn_output_dim = hidden_dim * (2 if bidirectional else 1)
            self.attention_W1 = nn.Linear(rnn_output_dim, attention_dim, bias=False)
            self.attention_w2 = nn.Linear(attention_dim, 1, bias=False)
            self.tanh = nn.Tanh()
            fc_input_dim = rnn_output_dim
        else:
            fc_input_dim = hidden_dim * (2 if bidirectional else 1)
        
        # Classification layers
        self.fc1 = nn.Linear(fc_input_dim, 150)
        self.fc2 = nn.Linear(150, num_classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
    
    def attention_forward(self, rnn_outputs, lengths):
        """Apply Bahdanau attention mechanism"""
        # rnn_outputs: (batch_size, seq_len, hidden_dim * num_directions)
        batch_size, seq_len, hidden_dim = rnn_outputs.shape
        
        # Compute attention scores
        attention_scores = self.attention_W1(rnn_outputs)  # (batch_size, seq_len, attention_dim)
        attention_scores = self.tanh(attention_scores)
        attention_scores = self.attention_w2(attention_scores).squeeze(-1)  # (batch_size, seq_len)
        
        # Apply mask for padding tokens
        mask = torch.arange(seq_len, device=rnn_outputs.device).expand(
            batch_size, seq_len) < lengths.unsqueeze(1)
        attention_scores.masked_fill_(~mask, float('-inf'))
        
        # Apply softmax
        attention_weights = torch.softmax(attention_scores, dim=1)  # (batch_size, seq_len)
        
        # Weighted sum
        attended_output = torch.sum(
            attention_weights.unsqueeze(-1) * rnn_outputs, dim=1
        )  # (batch_size, hidden_dim * num_directions)
        
        return attended_output
    
    def forward(self, x, lengths):
        # x: (batch_size, seq_len)
        # lengths: (batch_size,)
        
        embedded = self.embedding(x)  # (batch_size, seq_len, embedding_dim)
        
        # Pack sequences for efficiency
        packed = nn.utils.rnn.pack_padded_sequence(
            embedded, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        
        # RNN forward pass
        if self.rnn_type == 'LSTM':
            packed_output, (hidden, cell) = self.rnn(packed)
        else:
            packed_output, hidden = self.rnn(packed)
        
        # Unpack sequences
        rnn_output, _ = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)
        
        if self.use_attention:
            # Use attention mechanism
            pooled = self.attention_forward(rnn_output, lengths)
        else:
            # Use last hidden state
            if self.bidirectional:
                # Concatenate forward and backward hidden states
                hidden = hidden.view(self.num_layers, 2, -1, self.hidden_dim)
                forward_hidden = hidden[-1, 0, :, :]  # Last layer, forward direction
                backward_hidden = hidden[-1, 1, :, :]  # Last layer, backward direction
                pooled = torch.cat((forward_hidden, backward_hidden), dim=1)
            else:
                pooled = hidden[-1]  # Last layer
        
        # Classification layers
        x = self.relu(self.fc1(pooled))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x

# Training and Evaluation Functions

def train(model, dataloader, optimizer, criterion, clip_value=0.25):
    """Train the model for one epoch"""
    model.train()
    total_loss = 0
    all_predictions = []
    all_labels = []
    
    for batch_num, (texts, labels, lengths) in enumerate(dataloader):
        model.zero_grad()
        
        logits = model(texts, lengths)
        
        if isinstance(criterion, nn.BCEWithLogitsLoss):
            labels = labels.float()
            if logits.dim() > 1:
                logits = logits.squeeze(-1)
        
        loss = criterion(logits, labels)
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_value)
        
        optimizer.step()
        
        total_loss += loss.item()
        
        # Store predictions for metrics
        if isinstance(criterion, nn.BCEWithLogitsLoss):
            predictions = (torch.sigmoid(logits) > 0.5).long()
        else:
            predictions = torch.argmax(logits, dim=-1)
        
        all_predictions.extend(predictions.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    
    avg_loss = total_loss / len(dataloader)
    accuracy = accuracy_score(all_labels, all_predictions)
    f1 = f1_score(all_labels, all_predictions, average='weighted')
    
    return avg_loss, accuracy, f1

def evaluate(model, dataloader, criterion):
    """Evaluate the model"""
    model.eval()
    total_loss = 0
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for texts, labels, lengths in dataloader:
            logits = model(texts, lengths)
            
            if isinstance(criterion, nn.BCEWithLogitsLoss):
                labels = labels.float()
                if logits.dim() > 1:
                    logits = logits.squeeze(-1)
            
            loss = criterion(logits, labels)
            total_loss += loss.item()
            
            # Store predictions for metrics
            if isinstance(criterion, nn.BCEWithLogitsLoss):
                predictions = (torch.sigmoid(logits) > 0.5).long()
            else:
                predictions = torch.argmax(logits, dim=-1)
            
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    avg_loss = total_loss / len(dataloader)
    accuracy = accuracy_score(all_labels, all_predictions)
    f1 = f1_score(all_labels, all_predictions, average='weighted')
    cm = confusion_matrix(all_labels, all_predictions)
    
    return avg_loss, accuracy, f1, cm

def run_experiment(model_type='baseline', rnn_type='GRU', use_pretrained=True,
                   max_size=-1, min_freq=1, lr=1e-4, batch_size=10,
                   hidden_dim=150, num_layers=2, dropout=0.1,
                   bidirectional=False, use_attention=False,
                   epochs=5, seed=7052020):
    """Run a complete experiment"""
    
    # Set random seeds
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # Load datasets
    print("Loading datasets...")
    train_dataset = NLPDataset.from_file('data/sst_train_raw.csv')
    valid_dataset = NLPDataset.from_file('data/sst_valid_raw.csv', 
                                        train_dataset.text_vocab, 
                                        train_dataset.label_vocab)
    test_dataset = NLPDataset.from_file('data/sst_test_raw.csv',
                                       train_dataset.text_vocab,
                                       train_dataset.label_vocab)
    
    # Create dataloaders
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, 
                                 shuffle=True, collate_fn=pad_collate_fn)
    valid_dataloader = DataLoader(valid_dataset, batch_size=32, 
                                 shuffle=False, collate_fn=pad_collate_fn)
    test_dataloader = DataLoader(test_dataset, batch_size=32, 
                                shuffle=False, collate_fn=pad_collate_fn)
    
    # Load embeddings
    pretrained_embeddings = None
    if use_pretrained:
        pretrained_embeddings = load_embeddings(train_dataset.text_vocab, 
                                               'data/sst_glove_6b_300d.txt')
    
    # Initialize model
    vocab_size = len(train_dataset.text_vocab)
    
    if model_type == 'baseline':
        model = BaselineModel(vocab_size, pretrained_embeddings=pretrained_embeddings)
    else:  # RNN model
        model = RNNModel(vocab_size, hidden_dim=hidden_dim, num_layers=num_layers,
                        rnn_type=rnn_type, dropout=dropout, bidirectional=bidirectional,
                        pretrained_embeddings=pretrained_embeddings,
                        use_attention=use_attention)
    
    # Initialize criterion and optimizer
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # Training loop
    print(f"Starting training for {epochs} epochs...")
    for epoch in range(epochs):
        train_loss, train_acc, train_f1 = train(model, train_dataloader, optimizer, criterion)
        valid_loss, valid_acc, valid_f1, _ = evaluate(model, valid_dataloader, criterion)
        
        print(f"Epoch {epoch+1}: valid accuracy = {valid_acc:.3f}")
    
    # Final test evaluation
    test_loss, test_acc, test_f1, test_cm = evaluate(model, test_dataloader, criterion)
    print(f"Test accuracy = {test_acc:.3f}")
    
    return {
        'test_accuracy': test_acc,
        'test_f1': test_f1,
        'test_loss': test_loss,
        'confusion_matrix': test_cm
    }

# Example usage and hyperparameter search
if __name__ == "__main__":
    # Task 2: Baseline model
    print("=== Task 2: Baseline Model ===")
    baseline_results = run_experiment(model_type='baseline')
    
    # # Task 3: RNN models
    # print("\n=== Task 3: RNN Models ===")
    # for rnn_type in ['RNN', 'LSTM', 'GRU']:
    #     print(f"\n--- {rnn_type} Results ---")
    #     rnn_results = run_experiment(model_type='rnn', rnn_type=rnn_type)
    
    # # Task 4: Hyperparameter search examples
    # print("\n=== Task 4: Hyperparameter Search ===")
    
    # # Example: Different hidden dimensions
    # for hidden_dim in [64, 150, 256]:
    #     print(f"\n--- Hidden dim: {hidden_dim} ---")
    #     results = run_experiment(model_type='rnn', rnn_type='GRU', 
    #                            hidden_dim=hidden_dim)
    
    # # Example: With and without pretrained embeddings
    # print("\n--- Without pretrained embeddings ---")
    # results_no_pretrained = run_experiment(model_type='rnn', rnn_type='GRU',
    #                                      use_pretrained=False)
    
    #