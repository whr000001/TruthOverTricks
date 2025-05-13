import torch
import torch.nn as nn


class MyModel(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.linear_in = nn.Sequential(
            nn.Linear(1024, hidden_dim),
            nn.LeakyReLU()
        )
        self.cls = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, 2)
        )
        self.loss_fn = nn.CrossEntropyLoss()
        self.dropout = nn.Dropout(0.3)

    def forward(self, data):
        reps = self.dropout(self.linear_in(data['claims']))

        pred = self.cls(reps)
        loss = self.loss_fn(pred, data['labels'])

        return pred, loss, data['labels'], data['batch_size']
