import torch
from torch.utils.data import Dataset, Sampler


class MySampler(Sampler):
    def __init__(self, indices, shuffle):
        super().__init__(None)
        self.indices = indices
        if not torch.is_tensor(self.indices):
            self.indices = torch.tensor(self.indices, dtype=torch.long)
        self.shuffle = shuffle

    def __iter__(self):
        if self.shuffle:
            indices = self.indices[torch.randperm(self.indices.shape[0])]
        else:
            indices = self.indices
        for item in indices:
            yield item

    def __len__(self):
        return len(self.indices)


class MyDataset(Dataset):
    def __init__(self, claims, labels, extractive, abstractive, device):
        self.claims = claims
        self.labels = labels
        self.extractive = extractive
        self.abstractive = abstractive
        self.device = device

    def __len__(self):
        return len(self.claims)

    def __getitem__(self, index):
        return {
            'claim': self.claims[index],
            'label': self.labels[index],
            'extractive': self.extractive[index],
            'abstractive': self.abstractive[index]
        }

    def get_collate_fn(self):
        def collate_fn(batch):
            claims = []
            labels = []
            extractive = []
            abstractive = []
            for index, item in enumerate(batch):
                claims.append(item['claim'])
                labels.append(item['label'])
                extractive.append(item['extractive'])
                abstractive.append(item['abstractive'])
            claims = torch.stack(claims).to(self.device)
            labels = torch.stack(labels).to(self.device)
            extractive = torch.stack(extractive).to(self.device)
            abstractive = torch.stack(abstractive).to(self.device)
            return {
                'claims': claims,
                'labels': labels,
                'extractive': extractive,
                'abstractive': abstractive,
                'batch_size': len(batch)
            }
        return collate_fn
