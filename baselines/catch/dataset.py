import json

import torch
import numpy as np
from torch.utils.data import Dataset, Sampler
from transformers import RobertaTokenizer, BartTokenizer


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


class EncodedDataset(Dataset):
    def __init__(self, input_sents, input_labels,
                 max_sequence_length=512):
        self.input_sents = input_sents
        self.input_labels = input_labels

        self.encoder_tokenizer = RobertaTokenizer.from_pretrained('FacebookAI/roberta-base')
        self.decoder_tokenizer = BartTokenizer.from_pretrained('facebook/bart-base')

        self.max_sequence_length = max_sequence_length

    def __len__(self):
        return len(self.input_sents)

    def __getitem__(self, index):
        return self.input_sents[index], self.input_labels[index]

    def get_collate_fn(self, device):
        def collate_fn(batch):
            sents, labels = zip(*batch)
            input_ids = self.encoder_tokenizer.batch_encode_plus(sents, padding=True, truncation=True,
                                                                 max_length=self.max_sequence_length,
                                                                 return_tensors='pt').to(device)
            mask_ids = self.decoder_tokenizer.batch_encode_plus(sents, padding=True, truncation=True,
                                                                max_length=self.max_sequence_length,
                                                                return_tensors='pt').to(device)
            labels = torch.tensor(labels, dtype=torch.long).to(device)
            return input_ids, labels, mask_ids, len(batch)

        return collate_fn


