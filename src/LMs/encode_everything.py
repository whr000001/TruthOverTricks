import json
import os.path
import torch
from torch.utils.data import Dataset, DataLoader, Sampler
from transformers import DebertaV2Model, DebertaV2Tokenizer, AutoModel, AutoTokenizer
from tqdm import tqdm
from argparse import ArgumentParser


parser = ArgumentParser()
parser.add_argument('--lm', type=str)
args = parser.parse_args()


# here we employed bert and deberta as the representative encoder-based LMs
lm = args.lm
assert lm in ['bert', 'deberta']
if lm == 'bert':
    lm_path = 'google-bert/bert-large-uncased'
else:
    lm_path = 'microsoft/deberta-v3-large'


class MyDataset(Dataset):
    def __init__(self, data, device, max_length=512):
        labels = []
        input_seq = []
        for item in data:
            labels.append(item['label'])
            input_seq.append(item['claim'])
        labels = torch.tensor(labels, dtype=torch.long)
        self.labels = labels
        self.data = input_seq
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(lm_path)
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]

    def get_collate_fn(self):
        def collate_fn(batch):
            input_tensor = self.tokenizer.batch_encode_plus(batch, return_tensors='pt', padding=True,
                                                            max_length=self.max_length, truncation=True).to(self.device)
            return input_tensor
        return collate_fn


def encode(data):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset = MyDataset(data, device)
    loader = DataLoader(dataset, shuffle=False, collate_fn=dataset.get_collate_fn(), batch_size=32)
    encoder = AutoModel.from_pretrained(lm_path).to(device)
    with torch.no_grad():
        res = []
        pbar = tqdm(loader, leave=False)
        for batch in pbar:
            out = encoder(**batch)
            reps = out.last_hidden_state
            attention_mask = batch['attention_mask']
            reps = torch.einsum('ijk,ij->ijk', reps, attention_mask)
            reps = torch.sum(reps, dim=1)
            attention_mask = torch.sum(attention_mask, dim=1).unsqueeze(-1)
            reps = reps / attention_mask
            res.append(reps.to('cpu'))
        res = torch.cat(res, dim=0).detach().clone()
        labels = dataset.labels
    return res, labels


def main():
    if not os.path.exists('encoded_data'):
        os.mkdir('encoded_data')
    if not os.path.exists(f'encoded_data/{lm}'):
        os.mkdir(f'encoded_data/{lm}')

    # for original datasets, intrinsic setting
    rt_path = '../../datasets'
    datasets = os.listdir(rt_path)
    for dataset in datasets:
        data = json.load(open(f'{rt_path}/{dataset}'))
        save_path = f'encoded_data/{lm}/original_{dataset}'.replace('.json', '.pt')
        if os.path.exists(save_path):
            continue
        claims, labels = encode(data)
        torch.save([claims, labels], save_path)

    # for manipulated datasets, extrinsic setting
    dataset_types = ['sentiment', 'word-choice', 'tone', 'age', 'gender', 'llm-generation']
    for dataset_type in dataset_types:
        rt_path = f'../../extrinsic/data/{dataset_type}'
        datasets = os.listdir(rt_path)
        for dataset in datasets:
            data = json.load(open(f'{rt_path}/{dataset}'))
            save_path = f'encoded_data/{lm}/{dataset_type}_{dataset}'.replace('.json', '.pt')
            if os.path.exists(save_path):
                continue
            claims, labels = encode(data)
            torch.save([claims, labels], save_path)

    # for SMF
    dataset_types = ['summary', 'neutral', 'vanilla']
    for dataset_type in dataset_types:
        rt_path = f'../../SMF/data/{dataset_type}'
        datasets = os.listdir(rt_path)
        for dataset in datasets:
            data = json.load(open(f'{rt_path}/{dataset}'))
            save_path = f'encoded_data/{lm}/{dataset_type}_{dataset}'.replace('.json', '.pt')
            if os.path.exists(save_path):
                continue
            claims, labels = encode(data)
            torch.save([claims, labels], save_path)


if __name__ == '__main__':
    main()
