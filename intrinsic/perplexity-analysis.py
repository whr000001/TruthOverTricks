#  this file is used to analyze the perplexity of each dataset
from tqdm import tqdm
import torch
import json
import os
from evaluate import load


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = load('perplexity', module_type='metric')


def main():
    datasets = os.listdir('../datasets')
    for dataset in datasets:
        data = json.load(open(f'../datasets/{dataset}'))
        candidate = []
        for item in tqdm(data, desc=dataset, leave=False):
            claim = item['claim']
            candidate.append(claim)
        res = model.compute(predictions=candidate,
                            model_id='openai-community/gpt2',
                            device='cuda',
                            max_length=512)
        res = res['perplexities']
        json.dump(res, open(f'data/perplexity_{dataset}', 'w'))


if __name__ == '__main__':
    main()
