import json
import torch
from dataset import EncodedDataset, MySampler
from torch.utils.data import DataLoader
from model import CATCH
from argparse import ArgumentParser
import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score
import os


parser = ArgumentParser()
parser.add_argument('--hidden_dim', type=int, default=512)
parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--dataset', type=str)
args = parser.parse_args()

hidden_dim = args.hidden_dim
lr = args.lr
batch_size = args.batch_size
which_dataset = args.dataset
assert which_dataset in ['rumoureval', 'pheme', 'twitter15', 'twitter16', 'celebrityDataset', 'fakeNewsDataset',
                         'multilingual', 'politifact', 'gossipcop', 'tianchi', 'antivax', 'COCO', 'kaggle1', 'kaggle2',
                         'NQ', 'streaming']


def train_one_epoch(model, optimizer, loader):
    model.train()
    ave_loss = 0
    cnt = 0
    all_truth = []
    all_preds = []
    for batch in loader:
        optimizer.zero_grad()
        out, loss, truth, length = model(batch)
        loss.backward()
        optimizer.step()

        preds = out.argmax(-1).to('cpu')
        truth = truth.to('cpu')

        ave_loss += loss.item() * length
        cnt += length
        all_truth.append(truth)
        all_preds.append(preds)

    ave_loss /= cnt
    all_preds = torch.cat(all_preds, dim=0).numpy()
    all_truth = torch.cat(all_truth, dim=0).numpy()
    return ave_loss, accuracy_score(all_truth, all_preds), \
        f1_score(all_truth, all_preds, average='micro'), f1_score(all_truth, all_preds, average='macro')


@torch.no_grad()
def validate(model, loader, tmp=None):
    model.eval()

    ave_loss = 0
    cnt = 0
    all_truth = []
    all_preds = []
    all_score = []
    for batch in loader:
        out, loss, truth, length = model(batch)

        preds = out.argmax(-1).to('cpu')
        truth = truth.to('cpu')
        score = torch.softmax(out, dim=-1).to('cpu')

        ave_loss += loss.item() * length
        cnt += length
        all_truth.append(truth)
        all_preds.append(preds)
        all_score.append(score)

    ave_loss /= cnt
    all_preds = torch.cat(all_preds, dim=0).numpy().tolist()
    all_truth = torch.cat(all_truth, dim=0).numpy().tolist()
    all_score = torch.cat(all_score, dim=0).numpy().tolist()
    return ave_loss, accuracy_score(all_truth, all_preds), \
        f1_score(all_truth, all_preds, average='micro'), f1_score(all_truth, all_preds, average='macro'), \
        all_truth, all_preds, all_score


def obtain_shortcut_dataset(first, second):
    first_data = json.load(open(f'../../datasets/{first}.json'))
    second_data = json.load(open(f'../../datasets/{second}.json'))
    assert len(first_data) == len(second_data)
    indices = list(range(len(first_data)))
    train_size = int(len(indices) * 0.6)
    val_size = int(len(indices) * 0.2)
    train_indices = indices[:train_size]
    val_indices = indices[train_size:train_size + val_size]
    test_indices = indices[train_size + val_size:]
    if first == second:
        data = first_data
    else:
        data = []
        for index in range(len(first_data)):
            label = first_data[index]['label']
            if index in test_indices:
                each = first_data[index] if label == 0 else second_data[index]
            else:
                each = second_data[index] if label == 0 else first_data[index]
            data.append(each)
    claims, labels = [], []
    for item in data:
        claims.append(item['claim'])
        labels.append(item['label'])
    dataset = EncodedDataset(claims, labels)
    train_sampler = MySampler(train_indices, shuffle=True)
    val_sampler = MySampler(val_indices, shuffle=False)
    test_sampler = MySampler(test_indices, shuffle=False)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_loader = DataLoader(dataset, sampler=train_sampler,
                              batch_size=batch_size, collate_fn=dataset.get_collate_fn(device))
    val_loader = DataLoader(dataset, sampler=val_sampler,
                            batch_size=batch_size, collate_fn=dataset.get_collate_fn(device))
    test_loader = DataLoader(dataset, sampler=test_sampler,
                             batch_size=batch_size, collate_fn=dataset.get_collate_fn(device))
    return train_loader, val_loader, test_loader


def train(train_loader, val_loader, test_loader, device):
    model = CATCH().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)

    best_metrics = 0
    best_state = model.state_dict()
    for key, value in best_state.items():
        best_state[key] = value.clone()
    no_up_limits = 25
    no_up = 0
    pbar = range(100)
    # pbar = tqdm(range(100))
    for _ in pbar:
        train_loss, train_acc, train_micro, train_macro = train_one_epoch(model, optimizer, train_loader)
        # print('train', _, train_loss, train_acc, train_f1)
        val_loss, val_acc, val_micro, val_macro, _, _, _ = validate(model, val_loader)
        # print('val', _, val_acc, val_f1)
        if isinstance(pbar, tqdm):
            pbar.set_postfix({
                'train_loss': train_loss,
                'train_micro': train_micro,
                'train_macro': train_macro,
                'val_micro': val_micro,
                'val_macro': val_macro
            })
        if val_micro > best_metrics:
            best_metrics = val_micro
            for key, value in model.state_dict().items():
                best_state[key] = value.clone()
            no_up = 0
        else:
            no_up += 1
        if no_up >= no_up_limits:
            break
    model.load_state_dict(best_state)
    test_loss, test_acc, test_micro, test_macro, all_truth, all_preds, all_score = validate(model, test_loader)
    test_micro *= 100
    test_macro *= 100
    return test_micro, test_macro, all_truth, all_preds, all_score


def obtain(first, second):
    train_loader, val_loader, test_loader = obtain_shortcut_dataset(first, second)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    best_micro = 0
    best_macro, best_all_truth, best_all_preds, best_all_score = None, None, None, None
    for _ in range(5):
        micro, macro, all_truth, all_preds, all_score = train(train_loader, val_loader, test_loader, device)
        if micro > best_micro:
            best_micro = micro
            best_macro = macro
            best_all_truth = all_truth
            best_all_preds = all_preds
            best_all_score = all_score
    return best_micro, best_macro, best_all_truth, best_all_preds, best_all_score


def main():
    dataset_names = [which_dataset]

    pbar = tqdm(total=len(dataset_names), leave=False)
    pbar.set_description_str('original')
    res_path = f'res/{which_dataset}_original.json'
    if os.path.exists(res_path):
        res_map = json.load(open(res_path))
    else:
        res_map = {}
    for dataset_name in dataset_names:
        name = f'{dataset_name}'
        res = obtain(f'original/{dataset_name}', f'original/{dataset_name}')
        if name not in res_map:
            res_map[name] = res
        else:
            if res[0] > res_map[name][0]:
                res_map[name] = res
        json.dump(res_map, open(res_path, 'w'))
        pbar.update()

    candidates = [
        ['rewriting', 'paraphrase', 'open-ended'],
        ['positive', 'negative'],
        ['simple', 'complex'],
        ['formal', 'informal'],
        ['young', 'elder'],
        ['male', 'female']
    ]
    for index, category in enumerate(['llm-generation', 'sentiment', 'word-choice', 'tone', 'age', 'gender']):
        candidate = candidates[index]
        pbar = tqdm(total=len(dataset_names) * len(candidate) * len(candidate), leave=False)
        pbar.set_description_str(category)
        res_path = f'res/{which_dataset}_{category}.json'
        if os.path.exists(res_path):
            res_map = json.load(open(res_path))
        else:
            res_map = {}
        for dataset_name in dataset_names:
            for fake in candidate:
                for real in candidate:
                    name = f'{dataset_name}_{fake}_{real}'
                    res = obtain(f'{category}/{dataset_name}_{fake}', f'{category}/{dataset_name}_{real}')
                    if name not in res_map:
                        res_map[name] = res
                    else:
                        if res[0] > res_map[name][0]:
                            res_map[name] = res
                    json.dump(res_map, open(res_path, 'w'))
                    pbar.update()


if __name__ == '__main__':
    main()
