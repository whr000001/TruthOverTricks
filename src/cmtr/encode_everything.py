import json
import os.path
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModel, AutoTokenizer
from tqdm import tqdm


lm_path = 'google-bert/bert-large-uncased'


class MyDataset(Dataset):
    def __init__(self, data, device, max_length=512):
        self.data = data
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


def main():
    if not os.path.exists('representations'):
        os.mkdir('representations')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    categories = ['extractive', 'abstractive']
    for category in categories:
        path = f'{category}_out'
        files = os.listdir(path)
        files = [_.replace('.json', '') for _ in files]
        for file in files:
            data = json.load(open(f'{path}/{file}.json'))
            dataset = MyDataset(data, device)
            loader = DataLoader(dataset, shuffle=False, collate_fn=dataset.get_collate_fn(), batch_size=32)
            encoder = AutoModel.from_pretrained(lm_path).to(device)
            save_path = f'representations/{category}_{file}.pt'
            if os.path.exists(save_path):
                continue
            with torch.no_grad():
                res = []
                pbar = tqdm(loader, leave=False)
                pbar.set_description_str(file)
                for batch in pbar:
                    out = encoder(**batch)
                    reps = out.last_hidden_state
                    attention_mask = batch['attention_mask']
                    reps = torch.einsum('ijk,ij->ijk', reps, attention_mask)
                    reps = torch.sum(reps, dim=1)
                    attention_mask = torch.sum(attention_mask, dim=1).unsqueeze(-1)
                    reps = reps / attention_mask
                    res.append(reps.to('cpu'))
                res = torch.cat(res, dim=0)
            torch.save(res, save_path)


if __name__ == '__main__':
    main()
