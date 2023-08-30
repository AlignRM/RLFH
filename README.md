# RLFH: On-Policy Self-Alignment with Fine-grained Knowledge Feedback for Hallucination Mitigation

![GitHub stars](https://img.shields.io/github/stars/AlignRM/RLFH?style=social)
![Python Version](https://img.shields.io/badge/python-3.8+-blue)
![License](https://img.shields.io/github/license/AlignRM/RLFH)

This repository contains the official implementation of our paper [On-Policy Self-Alignment with Fine-grained Knowledge Feedback for Hallucination Mitigation](https://arxiv.org/abs/2406.12221).

## Overview

RLFH is an on-policy self-alignment approach that enables LLMs to actively explore their knowledge boundaries and self-correct through fine-grained feedback signals. Our method introduces a self-assessment framework where responses are automatically decomposed into atomic facts and evaluated against external knowledge sources, generating token-level dense rewards that enable precise optimization without human intervention.

## Installation

```bash
# Clone the repository
git clone https://github.com/wenxueru/RLFH.git
cd RLFH

# Install dependencies
pip install -e .
```

## Usage

The following script is an example for training **Qwen2.5-Instruct-7B** with our RLFH method:

```bash
export MODEL_PATH=Qwen/Qwen2.5-7B-Instruct
export SAVE_PATH=./models/Qwen2.5-7B-Instruct-RLFH
export PROMPT_DATA=data/hotpot_qa_train_1w.jsonl

# Distributed training settings
export REF_NUM_NODES=2
export REF_NUM_GPUS_PER_NODE=8
export CRITIC_NUM_NODES=2
export CRITIC_NUM_GPUS_PER_NODE=8
export ACTOR_NUM_NODES=2
export ACTOR_NUM_GPUS_PER_NODE=8
export VLLM_NUM_ENGINES=8
export VLLM_TENSOR_PARALLEL_SIZE=2
export COLOCATE_ALL_MODELS=True
export ENABLE_PREFIX_CACHEING=True
export VLLM_SYNC_WITH_RAY=True
export VLLM_GPU_MEMORY_UTILIZATION=0.6

# Training parameters
export GRANULARITY=token
export MICRO_ROLLOUT_BATCH_SIZE=8
export ROLLOUT_BATCH_SIZE=128
export MICRO_TRAIN_BATCH_SIZE=8
export TRAIN_BATCH_SIZE=128
export N_SAMPLES_PER_PROMPT=1
export FREEZING_ACTOR_STEPS=0
export SAVE_STEP=10

# Reward configuration
export VERIFICATION_MAP="{'Correct':0.45,'Hedged correct':0.35,'Vague':-1,'Hedged wrong':-1.5,'Wrong':-1.7}"
export INFORMATIVE_MAP="{'5':1.25,'4':1,'3':0.75,'2':0.1,'1':-0.2}"
export ACTOR_LR=3e-7
export CRITIC_LR=9e-6
export INFO_WEIGHT=1.2

bash scripts/train.sh
```

You can customize the training process by modifying these environment variables according to your requirements.

## Results

Our experiments demonstrate that RLFH significantly reduces hallucination rates across multiple benchmarks (HotpotQA, SQuADv2, and Biography) while maintaining or improving the informativeness of generated responses. The fine-grained token-level rewards enable more precise optimization compared to traditional approaches.

For detailed results and methodology, please refer to the [paper](https://arxiv.org/abs/2406.12221).

## Citation

If you find our work useful, please cite our paper:

```bibtex
@misc{wen2025onpolicyselfalignmentfinegrainedknowledge,
      title={On-Policy Self-Alignment with Fine-grained Knowledge Feedback for Hallucination Mitigation}, 
      author={Xueru Wen and Jie Lou and Xinyu Lu and Ji Yuqiu and Xinyan Guan and Yaojie Lu and Hongyu Lin and Ben He and Xianpei Han and Debing Zhang and Le Sun},
      year={2025},
      eprint={2406.12221},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2406.12221}, 
}
```

## Contact

If you have any questions related to the code, the paper, or copyright concerns, please contact:
- Email: `wenxueru2022@iscas.ac.cn`