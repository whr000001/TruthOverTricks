import os
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from argparse import ArgumentParser
import random
from tqdm import tqdm
import transformers

parser = ArgumentParser()
parser.add_argument('--gpus', type=str, default='0,1')
args = parser.parse_args()

gpus = args.gpus
os.environ['CUDA_VISIBLE_DEVICES'] = gpus
pipeline = transformers.pipeline(
    'text-generation',
    model='meta-llama/Meta-Llama-3-8B-Instruct',
    device_map="auto",
)
tokenizer = AutoTokenizer.from_pretrained('meta-llama/Meta-Llama-3-8B-Instruct')


def construct_length(text, max_length=1024):
    text = tokenizer.encode(text)
    text = text[:max_length]
    text = tokenizer.decode(text, skip_special_tokens=True)
    return text[:max_length]


@torch.no_grad()
def get_reply(prompt):
    terminators = [
        pipeline.tokenizer.eos_token_id,
        pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]
    messages = [
        {"role": "user", "content": prompt},
    ]
    outputs = pipeline(
        messages,
        max_new_tokens=1000,
        eos_token_id=terminators,
        do_sample=False,
        pad_token_id=pipeline.tokenizer.eos_token_id
    )
    return outputs[0]["generated_text"][-1]['content']


def main():
    datasets = os.listdir('../datasets')
    for dataset in datasets:
        dataset = dataset.replace('.json', '')
        data = json.load(open(f'../datasets/{dataset}.json'))
        pbar = tqdm(total=2 * len(data), leave=False)
        pbar.set_description_str(dataset)
        for profile in ['simple', 'complex']:
            save_path = f'data/word-choice/{dataset}_{profile}.json'
            if os.path.exists(save_path):
                continue
            res = []
            for item in data:
                claim = item['claim']
                claim = construct_length(claim)
                prompt = 'Given a passage, please rewrite it without any explanations. ' \
                         'The content should be the same. Make sure the word choice of the rewritten passage is {}. ' \
                         'The passage is: {}'.format(profile, claim)
                new_claim = get_reply(prompt)
                res.append({
                    'claim': new_claim,
                    'label': item['label']
                })
                pbar.update()
            json.dump(res, open(save_path, 'w'))


if __name__ == '__main__':
    main()
