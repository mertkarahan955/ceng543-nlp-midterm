# src/bert_model.py
import torch
import torch.nn as nn
from transformers import DistilBertModel

class DistilBertRNN(nn.Module):
    def __init__(self, bert_model_name='distilbert-base-uncased', hidden_dim=128, rnn_type='lstm', num_layers=1, num_classes=2, freeze_bert=True):
        super().__init__()
        self.bert = DistilBertModel.from_pretrained(bert_model_name)
        if freeze_bert:
            for p in self.bert.parameters():
                p.requires_grad = False
        emb_dim = self.bert.config.hidden_size
        self.rnn_type = rnn_type.lower()
        if self.rnn_type == 'lstm':
            self.rnn = nn.LSTM(emb_dim, hidden_dim, num_layers=num_layers, bidirectional=True, batch_first=True)
        else:
            self.rnn = nn.GRU(emb_dim, hidden_dim, num_layers=num_layers, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(hidden_dim*2, num_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden = outputs.last_hidden_state  # batch x seq_len x emb_dim
        out, h = self.rnn(last_hidden)
        if isinstance(h, tuple):
            h = h[0]
        forward = h[-2,:,:]; backward = h[-1,:,:]
        cat = torch.cat([forward, backward], dim=1)
        return self.fc(cat)
