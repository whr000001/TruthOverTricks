import os
import json
import torch
from summarizer import Summarizer
from tqdm import tqdm
from transformers import AutoConfig, AutoTokenizer, AutoModel


os.environ['TOKENIZERS_PARALLELISM'] = 'false'


@torch.no_grad()
def main():
    if not os.path.exists('extractive_out'):
        os.mkdir('extractive_out')
    model_path = 'google-bert/bert-large-uncased'
    custom_config = AutoConfig.from_pretrained(model_path)
    custom_config.output_hidden_states = True
    custom_tokenizer = AutoTokenizer.from_pretrained('google-bert/bert-large-uncased')
    custom_model = AutoModel.from_pretrained('google-bert/bert-large-uncased', config=custom_config)
    model = Summarizer(custom_model=custom_model, custom_tokenizer=custom_tokenizer)

    # for intrinsic
    rt_path = '../../datasets'
    datasets = os.listdir(rt_path)
    for dataset in datasets:
        data = json.load(open(f'{rt_path}/{dataset}'))
        save_path = f'extractive_out/original_{dataset}'
        if os.path.exists(save_path):
            continue
        pbar = tqdm(data, leave=False)
        pbar.set_description_str(dataset)
        res = []
        for item in pbar:
            res.append(model(item['claim'], ratio=0.4))
        json.dump(res, open(save_path, 'w'))

    # for intrinsic
    rt_path = '../../extrinsic/data'
    dataset_types = ['sentiment', 'word-choice', 'tone', 'age', 'gender', 'llm-generation']
    for dataset_type in dataset_types:
        type_path = f'{rt_path}/{dataset_type}'
        datasets = os.listdir(type_path)
        for dataset in datasets:
            data = json.load(open(f'{rt_path}/{dataset_type}/{dataset}'))
            save_path = f'extractive_out/{dataset_type}_{dataset}'
            if os.path.exists(save_path):
                continue
            pbar = tqdm(data, leave=False)
            pbar.set_description_str(dataset)
            res = []
            for item in pbar:
                res.append(model(item['claim'], ratio=0.4))
            json.dump(res, open(save_path, 'w'))

    # for SMF
    rt_path = '../../SMF/data'
    dataset_types = ['vanilla', 'neutral', 'summary']
    for dataset_type in dataset_types:
        type_path = f'{rt_path}/{dataset_type}'
        datasets = os.listdir(type_path)
        for dataset in datasets:
            data = json.load(open(f'{rt_path}/{dataset_type}/{dataset}'))
            save_path = f'extractive_out/{dataset_type}_{dataset}'
            if os.path.exists(save_path):
                continue
            pbar = tqdm(data, leave=False)
            pbar.set_description_str(dataset)
            res = []
            for item in pbar:
                res.append(model(item['claim'], ratio=0.4))
            json.dump(res, open(save_path, 'w'))


if __name__ == '__main__':
    main()
