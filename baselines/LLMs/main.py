import os
import torch
import json
import transformers
from tqdm import tqdm
from argparse import ArgumentParser
from transformers import AutoModelForCausalLM, AutoTokenizer


parser = ArgumentParser()
parser.add_argument('--gpus', type=str, default='0,1')
parser.add_argument('--llm', type=str)
args = parser.parse_args()

gpus = args.gpus
os.environ['CUDA_VISIBLE_DEVICES'] = gpus
llm = args.llm

assert llm in ['mistral', 'llama']
if llm == 'mistral':
    model = AutoModelForCausalLM.from_pretrained('mistralai/Mistral-7B-Instruct-v0.3',
                                                 device_map='auto', local_files_only=True)
    tokenizer = AutoTokenizer.from_pretrained('mistralai/Mistral-7B-Instruct-v0.3', local_files_only=True)
else:
    model = AutoModelForCausalLM.from_pretrained('meta-llama/Meta-Llama-3-8B-Instruct',
                                                 device_map='auto', local_files_only=True)
    tokenizer = AutoTokenizer.from_pretrained('meta-llama/Meta-Llama-3-8B-Instruct', local_files_only=True)


def construct_length(text, max_length=512):
    text = tokenizer.encode(text)
    text = text[:max_length]
    text = tokenizer.decode(text, skip_special_tokens=True)
    return text[:max_length]


@torch.no_grad()
def get_reply(prompt):
    messages = [
        {"role": "user", "content": prompt},
    ]
    inputs = tokenizer.apply_chat_template(messages, return_tensors="pt")
    inputs = inputs.to(model.device)

    outputs = model.generate(inputs,
                             max_new_tokens=1000,
                             do_sample=False,
                             return_dict_in_generate=True,
                             output_scores=True,
                             # temperature = 0.01,
                             pad_token_id=tokenizer.eos_token_id)
    transition_scores = model.compute_transition_scores(
        outputs.sequences, outputs.scores, normalize_logits=True
    )
    input_length = inputs.shape[1]
    generated_ids = outputs.sequences[:, input_length:]
    decoded = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
    return decoded[0], generated_ids[0].to('cpu').numpy().tolist(), transition_scores[0].to('cpu').numpy().tolist()


def main():
    rt_path = '../../datasets'
    dataset_types = ['original', 'sentiment', 'word-choice', 'tone', 'age', 'gender', 'llm-generation']
    for dataset_type in dataset_types:
        type_path = f'{rt_path}/{dataset_type}'
        if not os.path.isdir(type_path):
            continue
        datasets = os.listdir(type_path)
        for dataset in datasets:
            data = json.load(open(f'{rt_path}/{dataset_type}/{dataset}'))
            save_path = f'results/{llm}/{dataset_type}_{dataset}'
            if os.path.exists(save_path):
                continue
            pbar = tqdm(data, leave=False)
            pbar.set_description_str(dataset)
            res = []
            for item in pbar:
                claim = construct_length(item['claim'])
                prompt = claim + '\n'
                prompt += 'Please check the above claim true or false. Just output \'True\' or \'False\'. \n'
                res.append(get_reply(prompt))
            json.dump(res, open(save_path, 'w'))


if __name__ == '__main__':
    main()
