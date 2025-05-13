import os
import json
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model_path = 'tabularisai/robust-sentiment-analysis'
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path).to(device)


@torch.no_grad()
def predict_sentiment(text):
    inputs = tokenizer(text.lower(), return_tensors="pt", truncation=True, padding=True, max_length=512).to(device)
    outputs = model(**inputs)

    probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
    predicted_class = torch.argmax(probabilities, dim=-1).item()

    sentiment_map = {0: "Very Negative", 1: "Negative", 2: "Neutral", 3: "Positive", 4: "Very Positive"}
    return sentiment_map[predicted_class]


def main():
    datasets = os.listdir('../datasets/original')
    for dataset in datasets:
        data = json.load(open(f'../datasets/original/{dataset}'))
        res = []
        for item in tqdm(data, desc=dataset, leave=False):
            claim = item['claim']
            sentiment = predict_sentiment(claim)
            res.append(sentiment)
        json.dump(res, open(f'data/sentiment_{dataset}', 'w'))


if __name__ == '__main__':
    main()
