import html
import sqlite3
from dataclasses import dataclass
from functools import partial
from typing import Optional

import datasets
import jsonlines
from datasets import Dataset, IterableDataset, concatenate_datasets

from nltk import word_tokenize
from transformers import AutoTokenizer


class DocDB(object):
    """Sqlite backed document storage.

    Implements get_doc_text(doc_id).
    """
    SPECIAL_SEPARATOR = "####SPECIAL####SEPARATOR####"

    def __init__(self, db_path=None):
        self.db_path = db_path
        self.connection = sqlite3.connect(self.db_path, check_same_thread=False)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def path(self):
        """Return the path to the file that backs this database."""
        return self.path

    def close(self):
        """Close the connection to the database."""
        self.connection.close()

    def get_text_from_title(self, title):
        """Fetch the raw text of the doc for 'doc_id'."""
        cursor = self.connection.cursor()
        cursor.execute("SELECT text FROM documents WHERE title = ?", (title,))
        results = cursor.fetchall()
        results = [r for r in results]
        assert results is not None and len(results) == 1, f"`topic` in your data ({title}) is likely to be not a valid title in the DB."
        results = [{"title": title, "text": para} for para in results[0][0].split(self.SPECIAL_SEPARATOR)]
        assert len(results) > 0, f"`topic` in your data ({title}) is likely to be not a valid title in the DB."
        cursor.close()
        return results


@dataclass
class DatasetConfig:
    dataset_path: str = 'hotpot_qa'
    dataset_name: Optional[str] = None
    test_size: Optional[int] = None
    n_questions: Optional[int] = None
    db_path: Optional[str] = None
    split: Optional[str] = None
    template_name: Optional[str] = None
    seed: int = 0

    def __post_init__(self):
        for attr in ["split", "db_path", "dataset_name"]:
            if getattr(self, attr) == "":
                setattr(self, attr, None)


def format_qa_pair(
        question: str,
        answer: Optional[str] = None,
        tokenizer: AutoTokenizer = None,
) -> str:
    add_generation_prompt = True
    chat = [{"role": "user", "content": question}]
    if answer is not None:
        chat.append({"role": "system", "content": answer})
        add_generation_prompt = False
    return tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=add_generation_prompt)


def build_question(item, tokenizer=None):
    return {"prompt": format_qa_pair(item['question'], tokenizer=tokenizer)}


def get_db_materials(dataset, db_path: str, num_proc: int = 64):
    def get(item):
        titles = list(map(html.escape, item['titles']))
        extra_materials = []
        with DocDB(db_path) as db:
            for title in titles:
                try:
                    text = [r['text'] for r in db.get_text_from_title(title)]
                    extra_materials.append(''.join(text).replace("<s>", "").replace("</s>", ""))
                except AssertionError:
                    print(f"Could not find pape for {title}")
                    continue
        return {"extra_materials": extra_materials, "titles": titles}

    dataset = dataset.map(get, num_proc=num_proc)
    return dataset.filter(lambda i: len(i['extra_materials']) == len(i['titles']))


def load_hotpot_qa(dataset_path: str, dataset_name: str, db_path: str):
    dataset = datasets.load_dataset(dataset_path, dataset_name, trust_remote_code=True)
    dataset = concatenate_datasets([dataset['train'], dataset['validation']])
    dataset = dataset.filter(lambda i: i['question'].endswith('?') and len(word_tokenize(i['question'])) >= 5)

    def transform(item):
        context, titles = item['context'], item['titles']
        indices = [context['title'].index(title) for title in titles]
        materials = [''.join(context['sentences'][idx]).strip() for idx in indices]
        materials.append(f'The correct answer for \"{item["question"][:-1].replace("?", "")}\" is "{item["answer"]}".')
        return {"materials": materials, "titles": titles}

    def get_titles(item):
        supporting_facts = item['supporting_facts']
        titles = list(set(supporting_facts['title']))
        return {"titles": titles}

    dataset = dataset.map(get_titles)
    if db_path is not None:
        dataset = get_db_materials(dataset, db_path)
    return dataset.map(transform)


def load_squad_v2(dataset_path: str, db_path: str):
    dataset = datasets.load_dataset(dataset_path)
    dataset = concatenate_datasets([dataset['train'], dataset['validation']])
    dataset = dataset.map(lambda i: {"question_len": len(word_tokenize(i['question']))})
    dataset = dataset.filter(lambda i: i['question'].endswith('?') and i["question_len"] >= 5)
    dataset = Dataset.from_pandas(dataset.to_pandas().drop_duplicates(subset='title'))

    def transform(item):
        return {"materials": [item['context'].strip()] + [
            f'The correct answer for \"{item["question"][:-1].replace("?", "")}\" is "{item["answers"]["text"][0]}".'
            if len(item["answers"]["text"]) > 0
            else f'"{item["question"]}" has no correct answer.'
        ]}

    def get_titles(item):
        return {"titles": [html.unescape(item['title'])]}

    dataset = dataset.map(get_titles)
    if db_path is not None:
        dataset = get_db_materials(dataset, db_path, num_proc=1)
    return dataset.map(transform)


def load_bio(dataset_path: str, db_path: str):
    with jsonlines.open(dataset_path) as reader:
        dataset = Dataset.from_list(list(reader))

    dataset = dataset.rename_column("input", "question")
    dataset = dataset.map(lambda i: {"titles": [html.unescape(i["topic"])]})
    return get_db_materials(dataset, db_path, num_proc=1)


def load_dataset(dataset_config: DatasetConfig):
    dataset_name = dataset_config.dataset_name
    db_path = dataset_config.db_path if dataset_config.db_path else None
    load_func_map = {
        "squad_v2": partial(load_squad_v2, db_path=db_path),
        'bio': partial(load_bio, db_path=db_path),
        'hotpot_qa': partial(load_hotpot_qa, dataset_name=dataset_name, db_path=db_path),
    }
    for key in load_func_map.keys():
        if key in dataset_config.dataset_path:
            load_func = load_func_map[key]
            break
    else:
        load_func = partial(datasets.load_dataset, dataset_name=dataset_name)
    dataset = load_func(dataset_config.dataset_path)

    n_questions = dataset_config.n_questions
    test_size = dataset_config.test_size
    n_questions = n_questions + test_size if test_size and n_questions else n_questions
    if n_questions and n_questions > 0:
        if isinstance(dataset, IterableDataset):
            dataset = dataset.take(n_questions)
        else:
            dataset = dataset.select(list(range(n_questions)))

    if dataset_config.template_name is not None:
        tokenizer = AutoTokenizer.from_pretrained(dataset_config.template_name)
        dataset = dataset.map(partial(build_question, tokenizer=tokenizer))

    if test_size:
        dataset = dataset.train_test_split(test_size=test_size, seed=dataset_config.seed)

    if dataset_config.split:
        dataset = dataset[dataset_config.split]

    return dataset
