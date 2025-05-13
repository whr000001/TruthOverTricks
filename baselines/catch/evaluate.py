import os
import json


def main():
    dataset_names = ['rumoureval', 'pheme', 'twitter15', 'twitter16', 'celebrityDataset', 'fakeNewsDataset',
                     'multilingual', 'politifact', 'gossipcop', 'tianchi', 'antivax', 'COCO', 'kaggle1', 'kaggle2',
                     'NQ', 'streaming']
    for dataset_name in dataset_names:
        data = json.load(open(f'res/{dataset_name}_original.json'))
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
        for dataset_name in dataset_names:
            data = json.load(open(f'res/{dataset_name}_{category}.json'))
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
