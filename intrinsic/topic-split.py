import json
import os.path
import random
import numpy as np


def main():
    random.seed(20250101)
    dataset_names = ['rumoureval', 'pheme', 'twitter15', 'twitter16', 'celebrityDataset', 'fakeNewsDataset',
                     'multilingual', 'politifact', 'gossipcop', 'tianchi', 'antivax', 'COCO', 'kaggle1', 'kaggle2',
                     'NQ', 'streaming']
    marks = ['Sports', 'Arts, Culture, and Entertainment', 'Business and Finance', 'Health and Wellness',
             'Lifestyle and Fashion', 'Science and Technology', 'Politics', 'Crime']
    for dataset_name in dataset_names:
        topics = json.load(open(f'data/topic_{dataset_name}.json'))
        data = json.load(open(f'../datasets/{dataset_name}.json'))

        topic_labels = []
        for item in topics:
            assert item in marks
            topic_labels.append(int(item in marks[:4]))

        indices = list(range(len(data)))
        train_size = int(len(indices) * 0.6)
        val_size = int(len(indices) * 0.2)

        train_indices = indices[:train_size]
        val_indices = indices[train_size:train_size + val_size]
        test_indices = indices[train_size + val_size:]

        train_val_indices = []
        for index in train_indices + val_indices:
            if data[index]['label'] == 0 and topic_labels[index] == 0:
                train_val_indices.append(index)
            if data[index]['label'] == 1 and topic_labels[index] == 1:
                train_val_indices.append(index)
        filtered_test = []
        for index in test_indices:
            if data[index]['label'] == 0 and topic_labels[index] == 1:
                train_val_indices.append(index)
            if data[index]['label'] == 1 and topic_labels[index] == 0:
                filtered_test.append(index)
        assert len(train_val_indices) != 0
        random.shuffle(train_val_indices)
        random_indices = random.sample(train_indices + val_indices, k=len(train_val_indices))

        train_size = int(len(train_val_indices) * 0.8)
        train_indices = train_val_indices[:train_size]
        val_indices = train_val_indices[train_size:]
        random_train_indices = random_indices[:train_size]
        random_val_indices = random_indices[train_size:]

        save_path = f'splits/{dataset_name}'
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        json.dump(filtered_test, open(f'{save_path}/topic_filtered_test.json', 'w'))
        json.dump(train_indices, open(f'{save_path}/topic_train.json', 'w'))
        json.dump(val_indices, open(f'{save_path}/topic_val.json', 'w'))
        json.dump(random_train_indices, open(f'{save_path}/topic_random_train.json', 'w'))
        json.dump(random_val_indices, open(f'{save_path}/topic_random_val.json', 'w'))


if __name__ == '__main__':
    main()
