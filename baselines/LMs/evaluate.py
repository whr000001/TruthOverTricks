import os
import json


def main():
    dataset_names = ['rumoureval', 'pheme', 'twitter15', 'twitter16', 'celebrityDataset', 'fakeNewsDataset',
                     'politifact', 'gossipcop', 'tianchi', 'multilingual', 'antivax', 'COCO', 'kaggle1', 'kaggle2',
                     'NQ', 'streaming']
    lm = 'deberta'
    data = json.load(open(f'res/{lm}_original.json'))
    for dataset_name in dataset_names:
        print(data[dataset_name][0], end=' ')
    print()
    categories = ['llm-generation', 'sentiment', 'word-choice', 'tone', 'age', 'gender']
    candidates = [
        ['rewriting', 'paraphrase', 'open-ended'],
        ['positive', 'negative'],
        ['simple', 'complex'],
        ['formal', 'informal'],
        ['young', 'elder'],
        ['male', 'female']
    ]
    for category, candidate in zip(categories, candidates):
        data = json.load(open(f'res/{lm}_{category}.json'))
        for dataset_name in dataset_names:
            score, cnt = 0, 0
            for fake in candidate:
                for real in candidate:
                    name = f'{dataset_name}_{fake}_{real}'
                    if fake != real:
                        score += data[name][0]
                        cnt += 1
            score /= cnt
            print(score, end=' ')
        print()


if __name__ == '__main__':
    main()
