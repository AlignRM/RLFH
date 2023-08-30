from typing import Dict

import jsonlines
from jsonargparse import CLI
from tqdm import tqdm

from source.dataset import DatasetConfig, load_dataset


def main(
    dataset_configs: Dict = {
        "hotpot_qa": DatasetConfig(
            dataset_path="hotpotqa/hotpot_qa",
            dataset_name="distractor",
            n_questions=20000,
            test_size=256,
            split="test",
            db_path="../.cache/enwiki-20230401.db"
        ),
        "squad_v2": DatasetConfig(
            dataset_path="rajpurkar/squad_v2",
            db_path="../.cache/enwiki-20230401.db"
        ),
        "biography": DatasetConfig(
            dataset_path="repos/biography.jsonl",
            db_path="../.cache/enwiki-20230401.db"
        ),
    },
    output_path: str = "data",
):
    for dataset_name, dataset_config in dataset_configs.items():
        dataset = load_dataset(dataset_config)
        with jsonlines.open(f"{output_path}/{dataset_name}.jsonl", "w") as writer:
            writer.write_all(tqdm(dataset.to_list(), desc=f"Writing {dataset_name}"))
    hotpot_test_dataset = load_dataset(dataset_configs["hotpot_qa"])
    hotpot_dataset = load_dataset(
        DatasetConfig(
            dataset_path="hotpotqa/hotpot_qa",
            dataset_name="distractor",
            db_path="../.cache/enwiki-20230401.db",
        )
    )
    hotpot_test_questions = hotpot_test_dataset.to_dict()['question']
    hotpot_train_dataset = [
        item for item in hotpot_dataset.to_list()
        if item['question'] not in hotpot_test_questions
    ]
    hotpot_train_dataset_1w = hotpot_train_dataset[:10000]
    with jsonlines.open(f"{output_path}/hotpot_qa_train_1w.jsonl", "w") as writer:
        writer.write_all(tqdm(hotpot_train_dataset_1w, desc="Writing hotpot_train_1w"))


if __name__ == "__main__":
    CLI(main)
