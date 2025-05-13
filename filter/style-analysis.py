import os
import json
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import pipeline


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model_path = 'cffl/bert-base-styleclassification-subjective-neutral'
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)
pipe = pipeline('text-classification', model=model, tokenizer=tokenizer, device=device,
                truncation=True, padding=True, max_length=512)


@torch.no_grad()
def predict_style(text):
    res = pipe(text)
    return res[0]['label']


def main():
    datasets = os.listdir('../datasets/original')
    for dataset in datasets:
        data = json.load(open(f'../datasets/original/{dataset}'))
        res = []
        for item in tqdm(data, desc=dataset, leave=False):
            claim = item['claim']
            style = predict_style(claim)
            res.append(style)
        json.dump(res, open(f'data/style_{dataset}', 'w'))


if __name__ == '__main__':
    main()
