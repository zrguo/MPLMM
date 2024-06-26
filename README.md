# Multimodal Prompt Learning with Missing Modalities for Sentiment Analysis and Emotion Recognition

[ACL 2024 Main] Official PyTorch implementation of the paper "Multimodal Prompt Learning with Missing Modalities for Sentiment Analysis and Emotion Recognition"

## Introduction

The development of multimodal models has significantly advanced multimodal sentiment analysis and emotion recognition. However, in real-world applications, the presence of various missing modality cases often leads to a degradation in the model's performance. In this work, we propose a novel multimodal Transformer framework using prompt learning to address the issue of missing modalities. Our method introduces three types of prompts: generative prompts, missing-signal prompts, and missing-type prompts. These prompts enable the generation of missing modality features and facilitate the learning of intra- and inter-modality information. Through prompt tuning, we achieve a substantial reduction in the number of trainable parameters. Extensive experiments and ablation studies are conducted to demonstrate the effectiveness and robustness of our method, showcasing its ability to effectively handle missing modalities. 

![overall](overall.png)



## Getting Started

### Requirements

- Python >= 3.8, PyTorch >= 1.8.0


```
git clone https://github.com/zrguo/MPLMM.git
```

### Run the Code

1. Pre-train the model on CMU-MOSEI without prompts
```
mkdir pretrained
python main.py --dataset "mosei" --data_path "mosei path" --drop_rate 0 --name "./pretrained/mosei.pt"
```

2. Fine-tuning the pre-trained model on downstream datasets and get results

- Fine-tuning on CMU-MOSI

  ```
  python main.py --pretrained_model "./pretrained/mosei.pt" --dataset "mosi" --data_path "mosi path" --drop_rate 0.7 --name "mosi.pt"
  ```

- Fine-tuning on IEMOCAP

  ```
  python main.py --pretrained_model "./pretrained/mosei.pt" --dataset "iemocap" --data_path "iemocap path" --drop_rate 0.7 --name "iemocap.pt"
  ```

- Fine-tuning on CH-SIMS

  ```
  python main.py --pretrained_model "./pretrained/mosei.pt" --dataset "sims" --data_path "sims path" --drop_rate 0.7 --name "sims.pt"
  ```

  

## Acknowledgements

This code is based on the backbone [MulT](https://github.com/yaohungt/Multimodal-Transformer).