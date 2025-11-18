# Truth over Tricks
This is the official repository for the paper at NeurIPS 2025: [Truth over Tricks: Measuring and Mitigating Shortcut Learning in Misinformation Detection](https://arxiv.org/pdf/2506.02350)

We have also uploaded the codes and related resources at the [Google Drive](https://drive.google.com/drive/folders/1DI5ZmiD_3y2V8VKt7bMYqfYSTGHzoo-F?usp=drive_link). If you want to directly replicate our results or employ the original/manipulated/debiased data, we recommend downloading it directly.

## Baselines and Datasets
We have evaluated three types of existing misinformation detectors using 16 datasets (including 14 public datasets and 2 generated knowledge-intensive datasets).
- Encoder-based LMs
  - BERT
  - DeBERTA
- LLM-based Detectors
  - Mistral
  - Llama
- Debiasing Detectors
  - CATCH: [paper](https://arxiv.org/pdf/2308.02080)
  - DisC: [paper](https://proceedings.neurips.cc/paper_files/paper/2022/file/9e47a0bc530cc88b09b7670d2c130a29-Paper-Conference.pdf)
  - CMTR: [paper](https://arxiv.org/pdf/2204.01841)

Note that, for debiasing detectors, we have modified some minor details and hyperparameters to make them adaptive for our employed datasets.

If you want to employ the original versions, please refer to their own papers.

We have also uploaded our employed sampled datasets. (16 datasets, refer to our paper or codes for details)

If you employ our version, please describe it correctly and cite our paper and the original paper.

## Intrinsic
We evaluate four widely used misinformation indicators (sentiment, style, topic, perplexity) under intrinsic settings.

If you want to split the dataset by yourself, pls run:
```
cd intrinsic
python <perplexity, sentiment, style, topic>-analysis.py
python <perplexity, sentiment, style, topic>-split.py
```

## Extrinsic
We have designed three types of extrinsic injection: (i) vanilla/llm-generation (as baseline for comparison); (ii) explicit (sentiment, tone, and word-choice); and (iii) implicit (age and gender). 

If you want to generate the manipulated instances by yourself, pls run:
```
cd extrinsic
python <age, gender, llm-generation, sentiment, tone, word-choice>.py
```

## SMF
We have designed three prompts to generate debiased instances using LLMs: vanilla, neutral, and summary.

If you want to generate the instances of SMF by yourself, pls run:
```
cd SMF
python <neutralm, summary, vanilla>.py
```

## Evaluate misinformation detectors under TruthOverTricks

For every detector except LLM-based detectors, you could obtain the results by running:
```
cd src
cd <LMs, catch, cmtr, disc>
python <intrinsic, extrinsic, SMF>_train.py
```
Before running this, you may need to preprocess the datasets by running:
```
python encode_everything.py  # LMs
python abstractive.py  # cmtr
python extractive.py  # cmtr
python encode_everything.py  # cmtr
python preprocess  # disc
```
For LLM-based detectors, you could obtain the response of LLMs by running:
```
python response_everything.py --llm <mistral, llama>
```

Noted that for some detectors, you may need to determine the LMs or datasets by adding "--lm bert" or "--dataset twitter16"

## Citation
If you find our work interesting/helpful, please consider citing this paper
```
@inproceedings{
wan2025truth,
title={Truth over Tricks: Measuring and Mitigating Shortcut Learning in Misinformation Detection},
author={Herun Wan and Jiaying Wu and Minnan Luo and Zhi Zeng and Zhixiong Su},
booktitle={The Thirty-ninth Annual Conference on Neural Information Processing Systems},
year={2025},
url={https://openreview.net/forum?id=ngxGNQE1M2}
}
```

```
@article{wan2025truth,
  title={Truth over Tricks: Measuring and Mitigating Shortcut Learning in Misinformation Detection},
  author={Wan, Herun and Wu, Jiaying and Luo, Minnan and Zeng, Zhi and Su, Zhixiong},
  journal={arXiv preprint arXiv:2506.02350},
  year={2025}
}
```

## Question?
Feel free to open issues in this repository! Instead of emails, GitHub issues are much better at facilitating a conversation between you and our team to address your needs. You can also contact Herun Wan through `wanherun at stu.xjtu.edu.cn`.

## Updating

### 20251118
- We have uploaded the complete resources of TruthOverTricks, including codes and related data.
- We have provided a brief guideline to employ TruthOverTricks.

### 20251113
- We plan to refine this repository by December.

### 20251001
- Our paper has been accepted to the NeurIPS 2025!🥳🥳🥳

### Before
- We have uploaded related codes. However, it's missing a lot of details.
