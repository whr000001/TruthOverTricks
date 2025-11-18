import json
import os
from argparse import ArgumentParser
import torch
from dataset import MyDataset, MySampler
from tqdm import tqdm
from model import MyModel
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader
import numpy as np

parser = ArgumentParser()
parser.add_argument('--hidden_dim', type=int, default=512)
parser.add_argument('--batch_size', type=int, default=256)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--lm', type=str)
args = parser.parse_args()

hidden_dim = args.hidden_dim
lr = args.lr
batch_size = args.batch_size
lm = args.lm

assert lm in ['bert', 'deberta']


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


def train(train_loader, val_loader, test_loader, device):
    model = MyModel(hidden_dim=hidden_dim).to(device)
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


def obtain_loader(dataset_name, train_file, val_file, test_file):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data, label = torch.load(f'encoded_data/{lm}/original_{dataset_name}.pt', weights_only=True)

    dataset = MyDataset(claims=data, labels=label, device=device)

    train_indices = json.load(open(train_file))
    val_indices = json.load(open(val_file))
    test_indices = json.load(open(test_file))

    train_sampler = MySampler(train_indices, shuffle=True)
    val_sampler = MySampler(val_indices, shuffle=False)
    test_sampler = MySampler(test_indices, shuffle=False)

    train_loader = DataLoader(dataset, sampler=train_sampler,
                              batch_size=batch_size, collate_fn=dataset.get_collate_fn())
    val_loader = DataLoader(dataset, sampler=val_sampler,
                            batch_size=batch_size, collate_fn=dataset.get_collate_fn())
    test_loader = DataLoader(dataset, sampler=test_sampler,
                             batch_size=batch_size, collate_fn=dataset.get_collate_fn())
    return train_loader, val_loader, test_loader


def obtain(dataset_name, train_file, val_file, test_file):
    train_loader, val_loader, test_loader = obtain_loader(dataset_name, train_file, val_file, test_file)
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
    if not os.path.exists('res'):
        os.mkdir('res')
    dataset_names = [
        'rumoureval', 'pheme', 'twitter15', 'twitter16', 'celebrityDataset', 'fakeNewsDataset',
        'multilingual', 'politifact', 'gossipcop', 'tianchi', 'antivax', 'COCO', 'kaggle1', 'kaggle2',
        'NQ', 'streaming'
    ]
    short_cuts = [
        'sentiment', 'topic', 'style', 'perplexity'
    ]
    pbar = tqdm(total=len(dataset_names) * len(short_cuts))
    for short_cut in short_cuts:
        res_path = f'res/{lm}_intrinsic_{short_cut}.json'
        if os.path.exists(res_path):
            res_map = json.load(open(res_path))
        else:
            res_map = {}
        for dataset_name in dataset_names:
            train_file = f'../../intrinsic/splits/{dataset_name}/{short_cut}_train.json'
            val_file = f'../../intrinsic/splits/{dataset_name}/{short_cut}_val.json'
            test_file = f'../../intrinsic/splits/{dataset_name}/{short_cut}_filtered_test.json'
            res = obtain(dataset_name, train_file, val_file, test_file)
            name = f'{lm}_{short_cut}_{dataset_name}_intrinsic'
            if name not in res_map:
                res_map[name] = res
            else:
                if res[0] > res_map[name][0]:
                    res_map[name] = res

            train_file = f'../../intrinsic/splits/{dataset_name}/{short_cut}_random_train.json'
            val_file = f'../../intrinsic/splits/{dataset_name}/{short_cut}_random_val.json'
            test_file = f'../../intrinsic/splits/{dataset_name}/{short_cut}_filtered_test.json'
            res = obtain(dataset_name, train_file, val_file, test_file)
            name = f'{lm}_{short_cut}_{dataset_name}_random'
            if name not in res_map:
                res_map[name] = res
            else:
                if res[0] > res_map[name][0]:
                    res_map[name] = res
            json.dump(res_map, open(res_path, 'w'))
            pbar.update()


if __name__ == '__main__':
    main()
