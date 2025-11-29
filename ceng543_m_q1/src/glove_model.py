# src/glove_model.py
import torch
import torch.nn as nn
import torch.optim as optim

class BiRNNClassifier(nn.Module):
    def __init__(self, input_dim=300, hidden_dim=128, num_layers=1, rnn_type='lstm', num_classes=2, dropout=0.2):
        super().__init__()
        self.rnn_type = rnn_type.lower()
        # PyTorch RNN dropout only applies when num_layers > 1
        rnn_dropout = dropout if num_layers > 1 else 0.0
        if self.rnn_type == 'lstm':
            self.rnn = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers, bidirectional=True, batch_first=True, dropout=rnn_dropout)
        else:
            self.rnn = nn.GRU(input_dim, hidden_dim, num_layers=num_layers, bidirectional=True, batch_first=True, dropout=rnn_dropout)
        self.fc = nn.Linear(hidden_dim*2, num_classes)

    def forward(self, x):  # x: batch x seq_len x input_dim
        # pack padded sequences if you compute lengths (we used simple padding, so skipping pack)
        outputs, h = self.rnn(x)  # h: (num_layers*2, batch, hidden)
        # take last layer forward/backward
        if isinstance(h, tuple):  # LSTM returns (h, c)
            h = h[0]
        # h shape: (num_layers*2, batch, hidden_dim)
        # take last forward and last backward
        forward = h[-2, :, :]
        backward = h[-1, :, :]
        cat = torch.cat([forward, backward], dim=1)
        return self.fc(cat)

def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    for x, y in dataloader:
        x = x.to(device); y = y.to(device)
        logits = model(x)
        loss = criterion(logits, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * x.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == y).sum().item()
        total += y.size(0)
    return total_loss / total, correct / total

def eval_model(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0; correct = 0; total = 0
    with torch.no_grad():
        for x, y in dataloader:
            x = x.to(device); y = y.to(device)
            logits = model(x)
            loss = criterion(logits, y)
            total_loss += loss.item() * x.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)
    return total_loss / total, correct / total
