import os
from dataclasses import dataclass, field
from functools import partial
from typing import Any, Dict, Optional

import jsonlines
import numpy as np
import ray
from jsonargparse import CLI
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy
from tqdm import tqdm
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

from source.dataset import DatasetConfig, build_question, load_dataset

KEYS = {'question', 'materials', 'titles', 'extra_materials'}


def scheduling_strategy_fn(tensor_parallel_size):
    # One bundle per tensor parallel worker
    pg = ray.util.placement_group(
        [{"GPU": 1, "CPU": 1}] * tensor_parallel_size,
        strategy="STRICT_PACK",
    )
    return dict(scheduling_strategy=PlacementGroupSchedulingStrategy(
        pg, placement_group_capture_child_tasks=True))


@dataclass
class AnswerArguments:
    max_tokens: Optional[int] = field(
        default=1024,
        metadata={
            "help": "How many new tokens to be generated in maximum."
        },
    )
    n: Optional[int] = field(
        default=1,
        metadata={
            "help": "How many response run at one batch."
        },
    )
    temperature: Optional[float] = field(
        default=0.,
        metadata={
            "help": "Hyperparameter for random inference."
        },
    )
    top_p: float = field(
        default=1,
        metadata={
            "help": "Hyperparameter for random inference."
        },
    )


def main(
    data_path: Optional[str] = None,
    dataset_config: DatasetConfig = DatasetConfig(
        dataset_path="repos/biography.jsonl",
        template_name="Qwen/Qwen2.5-7B-Instruct",
        db_path="../cache/enwiki-20230401.db",
    ),
    model_path: str = "Qwen/Qwen2.5-7B-Instruct",
    output_path: str = "../Qwen2.5-7B-Instruct.jsonl",
    ans_args: AnswerArguments = AnswerArguments(),
    tensor_parallel_size: int = 1,
    num_instances: int = 1,
    batch_size: int = 8,
):

    class LLMPredictor:  # Create a class to do batch inference.
        def __init__(self):
            # Create an LLM.
            self.llm = LLM(
                model=model_path,
                tensor_parallel_size=tensor_parallel_size,
                trust_remote_code=True,
            )
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_path, trust_remote_code=True,
            )
            self.kwargs = ans_args.__dict__

        def __call__(self, batch):
            questions = batch['prompt']
            answers = self.llm.generate(questions, sampling_params=SamplingParams(**self.kwargs, skip_special_tokens=True))
            answers = [[a.text for a in ans.outputs] for ans in answers]
            new_batch = {
                key: batch[key] if not isinstance(batch[key][0], np.ndarray) else [arr.tolist() for arr in batch[key]]
                for key in KEYS if key in batch
            }
            new_batch['answers'] = answers
            return new_batch

    if data_path is not None:
        dataset = []
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        with jsonlines.open(data_path) as reader:
            for item in reader:
                item.update(build_question(item, tokenizer=tokenizer))
                dataset.append(item)
    else:
        dataset_config.template_name = model_path
        dataset = load_dataset(dataset_config).to_list()
    ds = ray.data.from_items(dataset)

    resources_kwarg: Dict[str, Any] = {}
    if tensor_parallel_size == 1:
        # For tensor_parallel_size == 1, we simply set num_gpus=1.
        resources_kwarg["num_gpus"] = 1
    else:
        # Otherwise, we have to set num_gpus=0 and provide
        # a function that will create a placement group for
        # each instance.
        resources_kwarg["num_gpus"] = 0
        resources_kwarg["ray_remote_args_fn"] = partial(
            scheduling_strategy_fn,
            tensor_parallel_size=tensor_parallel_size
        )
    ds = ds.map_batches(
        LLMPredictor,
        # Set the concurrency to the number of LLM instances.
        concurrency=(num_instances, 1024),
        # Specify the batch size for inference.
        batch_size=batch_size,
        **resources_kwarg,
    )
    dataset = list(ds.iter_rows())

    if os.path.isdir(output_path):
        output_path = os.path.join(output_path, 'qa.jsonl')
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with jsonlines.open(output_path, 'w') as writer:  # type: ignore
        writer.write_all(tqdm(dataset, desc="Writing to file", position=1))


if __name__ == '__main__':
    CLI(main)
