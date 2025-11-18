import json
import os.path
import random
import numpy as np


def main():
    random.seed(20250101)
    dataset_names = ['rumoureval', 'pheme', 'twitter15', 'twitter16', 'celebrityDataset', 'fakeNewsDataset',
                     'multilingual', 'politifact', 'gossipcop', 'tianchi', 'antivax', 'COCO', 'kaggle1', 'kaggle2',
                     'NQ', 'streaming']
    for dataset_name in dataset_names:
        perplexities = json.load(open(f'data/perplexity_{dataset_name}.json'))
        data = json.load(open(f'../datasets/{dataset_name}.json'))

        perplexity_median = np.median(perplexities)  # binary classification of the perplexity
        perplexity_labels = [_ < perplexity_median for _ in perplexities]

        indices = list(range(len(data)))
        train_size = int(len(indices) * 0.6)
        val_size = int(len(indices) * 0.2)

        # split for normal setting
        train_indices = indices[:train_size]
        val_indices = indices[train_size:train_size + val_size]
        test_indices = indices[train_size + val_size:]

        # make the environment of train-val and test sets different
        train_val_indices = []
        for index in train_indices + val_indices:
            if data[index]['label'] == 0 and perplexity_labels[index] == 0:
                train_val_indices.append(index)
            if data[index]['label'] == 1 and perplexity_labels[index] == 1:
                train_val_indices.append(index)
        filtered_test = []
        for index in test_indices:
            if data[index]['label'] == 0 and perplexity_labels[index] == 1:
                train_val_indices.append(index)
            if data[index]['label'] == 1 and perplexity_labels[index] == 0:
                filtered_test.append(index)

        assert len(train_val_indices) != 0
        random.shuffle(train_val_indices)
        random_indices = random.sample(train_indices + val_indices, k=len(train_val_indices))  # for comparison

        train_size = int(len(train_val_indices) * 0.8)
        train_indices = train_val_indices[:train_size]
        val_indices = train_val_indices[train_size:]  # re-split
        random_train_indices = random_indices[:train_size]
        random_val_indices = random_indices[train_size:]

        save_path = f'splits/{dataset_name}'
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        json.dump(filtered_test, open(f'{save_path}/perplexity_filtered_test.json', 'w'))
        json.dump(train_indices, open(f'{save_path}/perplexity_train.json', 'w'))
        json.dump(val_indices, open(f'{save_path}/perplexity_val.json', 'w'))
        json.dump(random_train_indices, open(f'{save_path}/perplexity_random_train.json', 'w'))
        json.dump(random_val_indices, open(f'{save_path}/perplexity_random_val.json', 'w'))


if __name__ == '__main__':
    main()
