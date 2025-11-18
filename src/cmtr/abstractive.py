import os
import json
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, BartForConditionalGeneration


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model_path = 'sshleifer/distilbart-cnn-12-6'
model = BartForConditionalGeneration.from_pretrained(model_path).to(device)
tokenizer = AutoTokenizer.from_pretrained(model_path)


@torch.no_grad()
def abstract(text):
    inputs = tokenizer([text], max_length=1024, return_tensors='pt', truncation=True).to(device)

    summary_ids = model.generate(inputs['input_ids'], do_sample=True, top_k=100, top_p=0.95,
                                 min_length=5, max_length=512)
    return tokenizer.batch_decode(summary_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]


@torch.no_grad()
def main():
    if not os.path.exists('abstractive_out'):
        os.mkdir('abstractive_out')

    # for intrinsic
    rt_path = '../../datasets'
    datasets = os.listdir(rt_path)
    for dataset in datasets:
        data = json.load(open(f'{rt_path}/{dataset}'))
        save_path = f'abstractive_out/original_{dataset}'
        if os.path.exists(save_path):
            continue
        pbar = tqdm(data, leave=False)
        pbar.set_description_str(dataset)
        res = []
        for item in pbar:
            res.append(abstract(item['claim']))
        json.dump(res, open(save_path, 'w'))

    # for extrinsic
    rt_path = '../../extrinsic/data'
    dataset_types = ['sentiment', 'word-choice', 'tone', 'age', 'gender', 'llm-generation']
    for dataset_type in dataset_types:
        type_path = f'{rt_path}/{dataset_type}'
        datasets = os.listdir(type_path)
        for dataset in datasets:
            data = json.load(open(f'{rt_path}/{dataset_type}/{dataset}'))
            save_path = f'abstractive_out/{dataset_type}_{dataset}'
            if os.path.exists(save_path):
                continue
            pbar = tqdm(data, leave=False)
            pbar.set_description_str(dataset)
            res = []
            for item in pbar:
                res.append(abstract(item['claim']))
            json.dump(res, open(save_path, 'w'))

    # for SMF
    rt_path = '../../SMF/data'
    dataset_types = ['vanilla', 'neutral', 'summary']
    for dataset_type in dataset_types:
        type_path = f'{rt_path}/{dataset_type}'
        datasets = os.listdir(type_path)
        for dataset in datasets:
            data = json.load(open(f'{rt_path}/{dataset_type}/{dataset}'))
            save_path = f'abstractive_out/{dataset_type}_{dataset}'
            if os.path.exists(save_path):
                continue
            pbar = tqdm(data, leave=False)
            pbar.set_description_str(dataset)
            res = []
            for item in pbar:
                res.append(abstract(item['claim']))
            json.dump(res, open(save_path, 'w'))


if __name__ == '__main__':
    main()

