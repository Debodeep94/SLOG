import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
from sklearn.preprocessing import MinMaxScaler
import torch.nn.functional as F

class TanhAttention(nn.Module):
    def __init__(self, hidden_size,num_labels, dropout=0.5):
        super(TanhAttention, self).__init__()
        self.attn1 = nn.Linear(hidden_size, hidden_size // 2)
        self.attn2 = nn.Linear(hidden_size // 2, 1, bias=False)
        self.dropout = nn.Dropout(p=dropout)
        self.fc = nn.Linear(hidden_size, num_labels)

    def forward(self, output, mask):
        attn1 = nn.Tanh()(self.attn1(output))
        attn2 = self.attn2(attn1).squeeze(-1)
        attn = F.softmax(torch.add(attn2, mask), dim=1)

        h = output.transpose(1, 2).matmul(attn.unsqueeze(2)).squeeze(2)
        y_hat = self.fc(self.dropout(h))

        return y_hat


class LSTM_Attn(nn.Module):
    def __init__(self,input_size, hidden_size,num_labels, num_classes=14):

        super(LSTM_Attn, self).__init__()

        #self.embed = nn.Embedding.from_pretrained(torch.load('target_embedding_weights.pth'), freeze=True)

        self.rnn = nn.LSTM(input_size=input_size, hidden_size=hidden_size, batch_first=True, bidirectional=True)

        self.attns = nn.ModuleList([TanhAttention(hidden_size*2,num_labels) for i in range(num_classes)])

    def generate_pad_mask(self, batch_size, caption_lengths, max_len):

        mask = torch.full((batch_size, max_len), fill_value=float('-inf'), dtype=torch.float, device='cuda')
        for ind, cap_len in enumerate(caption_lengths):
            mask[ind][:cap_len] = 0

        return mask

    def forward(self, encoded_captions, caption_lengths): #here x is encoded captions
        #x = self.embed(encoded_captions)
        x=encoded_captions.to(torch.float32)
        #print(x)
        batch_size = x.size(0)
        max_len = encoded_captions.size(1)
        padding_mask = self.generate_pad_mask(batch_size, caption_lengths, max_len)

        output, (_, _) = self.rnn(x)


        y_hats = [attn(output, padding_mask) for attn in self.attns]
        y_hats = torch.stack(y_hats, dim=1)

        return y_hats


