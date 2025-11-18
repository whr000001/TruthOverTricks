import torch
import torch.nn as nn


class MyModel(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.linear_claim = nn.Sequential(
            nn.Linear(1024, hidden_dim),
            nn.LeakyReLU()
        )
        self.linear_extractive = nn.Sequential(
            nn.Linear(1024, hidden_dim),
            nn.LeakyReLU()
        )
        self.linear_abstractive = nn.Sequential(
            nn.Linear(1024, hidden_dim),
            nn.LeakyReLU()
        )
        self.dropout = nn.Dropout(0.3)
        self.cls = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, 2)
        )
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, data):
        claim = self.linear_claim(data['claims'])
        extractive = self.linear_claim(data['extractive'])
        abstractive = self.linear_claim(data['abstractive'])

        reps = torch.cat([claim, extractive, abstractive], dim=-1)
        reps = self.dropout(reps)

        pred = self.cls(reps)
        loss = self.loss_fn(pred, data['labels'])

        return pred, loss, data['labels'], data['batch_size']

