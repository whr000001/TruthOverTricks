import os
import json
import torch
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
tokenizer = AutoTokenizer.from_pretrained('google-bert/bert-large-uncased')
model = AutoModel.from_pretrained('google-bert/bert-large-uncased').to(device)


@torch.no_grad()
def obtain_graph(text):
    inputs = tokenizer([text], max_length=512, return_tensors='pt', truncation=True).to(device)
    out = model(**inputs)
    x = out.last_hidden_state.squeeze(0).to('cpu')
    row, col = [], []
    node_cnt = x.shape[0]
    for i in range(node_cnt):
        for j in range(i-10, i+10):
            if i != j and 0 <= j < node_cnt:
                row.append(i)
                col.append(j)
                row.append(j)
                col.append(i)
    edge_index = torch.tensor([row, col], dtype=torch.long)
    return [x, edge_index]


def main():
    if not os.path.exists('text_graphs'):
        os.mkdir('text_graphs')

    # for intrinsic
    rt_path = '../../datasets'
    datasets = os.listdir(rt_path)
    for dataset in datasets:
        data = json.load(open(f'{rt_path}/{dataset}'))
        save_path = f'text_graphs/original_{dataset}'.replace('.json', '.pt')
        if os.path.exists(save_path):
            continue
        pbar = tqdm(data, leave=False)
        pbar.set_description_str(dataset)
        res = []
        for item in pbar:
            claim = item['claim']
            text_graph = obtain_graph(claim)
            res.append(text_graph)
        torch.save(res, save_path)

    # for extrinsic
    rt_path = '../../extrinsic/data'
    dataset_types = ['sentiment', 'word-choice', 'tone', 'age', 'gender', 'llm-generation']
    for dataset_type in dataset_types:
        type_path = f'{rt_path}/{dataset_type}'
        datasets = os.listdir(type_path)
        for dataset in datasets:
            data = json.load(open(f'{rt_path}/{dataset_type}/{dataset}'))
            save_path = f'text_graphs/{dataset_type}_{dataset}'.replace('.json', '.pt')
            if os.path.exists(save_path):
                continue
            pbar = tqdm(data, leave=False)
            pbar.set_description_str(dataset)
            res = []
            for item in pbar:
                claim = item['claim']
                text_graph = obtain_graph(claim)
                res.append(text_graph)
            torch.save(res, save_path)

    # for SMF
    rt_path = '../../SMF/data'
    dataset_types = ['vanilla', 'summary', 'neutral']
    for dataset_type in dataset_types:
        type_path = f'{rt_path}/{dataset_type}'
        datasets = os.listdir(type_path)
        for dataset in datasets:
            data = json.load(open(f'{rt_path}/{dataset_type}/{dataset}'))
            save_path = f'text_graphs/{dataset_type}_{dataset}'.replace('.json', '.pt')
            if os.path.exists(save_path):
                continue
            pbar = tqdm(data, leave=False)
            pbar.set_description_str(dataset)
            res = []
            for item in pbar:
                claim = item['claim']
                text_graph = obtain_graph(claim)
                res.append(text_graph)
            torch.save(res, save_path)


if __name__ == '__main__':
    main()
