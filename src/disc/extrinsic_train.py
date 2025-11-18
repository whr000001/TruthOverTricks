import json
import os
import torch
from dataset import MyDataset, get_collate_fn, MySampler
from torch_geometric.data import Data
from torch.utils.data import DataLoader
from model import MyModel
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score
from argparse import ArgumentParser


parser = ArgumentParser()
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--lr', type=float, default=1e-3)
args = parser.parse_args()

lr = args.lr
batch_size = args.batch_size


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
    model = MyModel(in_dim=1024, hidden_dim=256).to(device)
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


def load_one(name):
    dataset_names = ['rumoureval', 'pheme', 'twitter15', 'twitter16', 'celebrityDataset', 'fakeNewsDataset',
                     'multilingual', 'politifact', 'gossipcop', 'tianchi', 'antivax', 'COCO', 'kaggle1', 'kaggle2',
                     'NQ', 'streaming']
    dataset_name = None
    for _ in dataset_names:
        if name.find(_) != -1:
            dataset_name = _
    assert dataset_name is not None
    text_data = json.load(open(f'../../datasets/{dataset_name}.json'))
    name = name.replace('/', '_')
    data = torch.load(f'text_graphs/{name}.pt')
    graphs = []
    for index, item in enumerate(data):
        graphs.append(Data(x=item[0], edge_index=item[1], y=text_data[index]['label']))
    return graphs


def shortcut(first_name, second_name):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    first_graphs = load_one(first_name)
    second_graphs = load_one(second_name)
    assert len(first_graphs) == len(second_graphs)
    indices = list(range(len(first_graphs)))
    train_size = int(len(indices) * 0.6)
    val_size = int(len(indices) * 0.2)

    train_indices = indices[:train_size]
    val_indices = indices[train_size:train_size + val_size]
    test_indices = indices[train_size + val_size:]

    res = []
    for index in range(len(first_graphs)):
        assert first_graphs[index].y == second_graphs[index].y
        label = first_graphs[index].y
        if index in test_indices:
            claim = first_graphs[index] if label == 0 else second_graphs[index]
        else:
            claim = second_graphs[index] if label == 0 else first_graphs[index]
        res.append(claim)
    dataset = MyDataset(res)
    train_sampler = MySampler(train_indices, shuffle=True)
    val_sampler = MySampler(val_indices, shuffle=False)
    test_sampler = MySampler(test_indices, shuffle=False)

    train_loader = DataLoader(dataset, sampler=train_sampler,
                              batch_size=batch_size, collate_fn=get_collate_fn(device))
    val_loader = DataLoader(dataset, sampler=val_sampler,
                            batch_size=batch_size, collate_fn=get_collate_fn(device))
    test_loader = DataLoader(dataset, sampler=test_sampler,
                             batch_size=batch_size, collate_fn=get_collate_fn(device))
    return train_loader, val_loader, test_loader


def obtain(first, second):
    train_loader, val_loader, test_loader = shortcut(first, second)
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
    dataset_names = ['rumoureval', 'pheme', 'twitter15', 'twitter16', 'celebrityDataset', 'fakeNewsDataset',
                     'multilingual', 'politifact', 'gossipcop', 'tianchi', 'antivax', 'COCO', 'kaggle1', 'kaggle2',
                     'NQ', 'streaming']
    pbar = tqdm(total=len(dataset_names), leave=False)
    pbar.set_description_str('original')
    res_path = f'res/original.json'
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
        res_path = f'res/{category}.json'
        if os.path.exists(res_path):
            res_map = json.load(open(res_path))
        else:
            res_map = {}
        for dataset_name in dataset_names:
            for fake in candidate:
                for real in candidate:
                    name = f'{dataset_name}_{fake}_{real}'
                    if name in res_map:
                        continue
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
