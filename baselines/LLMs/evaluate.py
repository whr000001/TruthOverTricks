import json
from sklearn.metrics import accuracy_score


llm = 'llama'


def identify(text):
    text = text.lower()
    false_index = text.find('false')
    true_index = text.find('true')
    false_index = len(text) + 1 if false_index == -1 else false_index
    true_index = len(text) + 1 if true_index == -1 else true_index
    return int(false_index < true_index)


def shortcut(first, second):
    first_data = json.load(open(f'../../datasets/{first}.json'))
    second_data = json.load(open(f'../../datasets/{second}.json'))
    assert len(first_data) == len(second_data)
    indices = list(range(len(first_data)))
    train_size = int(len(indices) * 0.6)
    val_size = int(len(indices) * 0.2)

    train_indices = indices[:train_size]
    val_indices = indices[train_size:train_size + val_size]
    test_indices = indices[train_size + val_size:]

    first_res_path = first.replace('/', '_')
    second_res_path = second.replace('/', '_')

    first_res = json.load(open(f'results/{llm}/{first_res_path}.json'))
    second_res = json.load(open(f'results/{llm}/{second_res_path}.json'))
    assert len(first_res) == len(second_res)

    data = []
    res = []
    for index in range(len(first_data)):
        assert first_data[index]['label'] == second_data[index]['label']
        label = first_data[index]['label']
        if index in test_indices:
            each = first_data[index] if label == 0 else second_data[index]
            each_res = first_res[index] if label == 0 else second_res[index]
        else:
            each = second_data[index] if label == 0 else first_data[index]
            each_res = second_res[index] if label == 0 else first_res[index]
        data.append(each)
        res.append(each_res)

    all_preds = []
    all_label = []
    for index in test_indices:
        preds = identify(res[index][0])
        label = data[index]['label']

        all_preds.append(preds)
        all_label.append(label)

    return accuracy_score(all_label, all_preds) * 100


def main():
    dataset_names = ['rumoureval', 'pheme', 'twitter15', 'twitter16', 'celebrityDataset', 'fakeNewsDataset',
                     'multilingual', 'politifact', 'gossipcop', 'tianchi', 'antivax', 'COCO', 'kaggle1', 'kaggle2',
                     'NQ', 'streaming']
    for dataset in dataset_names:
        res = shortcut(f'original/{dataset}', f'original/{dataset}')
        print(res, end=' ')
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
            score, cnt = 0, 0
            for fake in candidate:
                for real in candidate:
                    res = shortcut(f'{category}/{dataset_name}_{fake}', f'{category}/{dataset_name}_{real}')
                    if fake != real:
                        score += res
                        cnt += 1
            score /= cnt
            print(score, end=' ')
        print()


if __name__ == '__main__':
    main()
