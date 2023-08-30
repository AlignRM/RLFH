import bisect
import logging
from dataclasses import dataclass, field
from functools import partial
from math import log
from typing import Dict, List, Literal, Optional, Union

import numpy as np
import pylcs
from torch import Tensor
from tqdm import tqdm

from source.utils import div, lcs_sequence_idx, mean

from .annotate import Annotator, AnnotatorConfig

logger = logging.getLogger(__name__)


@dataclass
class RewardConfig:
    granularity: Literal["paragraph", "sentence", "token"] = "token"
    verification_map: dict = field(
        default_factory=lambda: {
            'Correct': 1,
            'Hedged correct': 0.5,
            'Vague': -1,
            'Hedged wrong': -1.5,
            'Wrong': -2,
        }
    )
    informative_map: dict = field(
        default_factory=lambda: {
            '5': 1.3,
            '4': 1.2,
            '3': 1.1,
            '2': 1,
            '1': -0.1,
        }
    )
    truth_weight: float = 1.
    info_weight: float = 1.
    epsilon: float = -0.9
    mu: float = 1

    annotator_config: AnnotatorConfig = AnnotatorConfig()


class Reward:
    def __init__(self, reward_config: RewardConfig):
        self.config = reward_config
        self.annotator = Annotator(reward_config.annotator_config)

    def reward_fn(
        self,
        outputs: List[str],
        questions: List[str],
        tokenizer=None,
        titles: List[List[str]] = None,
        materials: Optional[List[List[str]]] = None,
        extra_materials: Optional[List[List[str]]] = None,
        sample_outputs: Tensor = None,
        default_for_none: Optional[float] = 0,
        granularity: Optional[Literal["paragraph", "sentence", "token"]] = None,
        **kwargs
    ):
        questions = [ques.strip() for ques in questions]
        answers = [ans.strip() for ans in outputs]

        all_labels = self.annotator.annotate(
            answers=answers,
            questions=questions,
            titles=titles,
            materials=materials,
            extra_materials=extra_materials
        )
        all_scores = self.labels_to_rewards(
            sample_outputs, all_labels,
            tokenizer=tokenizer,
            default_for_none=default_for_none,
            granularity=granularity or self.config.granularity,
        )
        all_metrics = self.labels_to_metrics(all_labels)
        return all_scores, all_labels, all_metrics

    def labels_to_rewards(
        self,
        sample_outputs: Tensor = None,
        labels: List[dict] = None,
        tokenizer=None,
        default_for_none: Optional[float] = 0,
        granularity: Optional[Literal["paragraph", "sentence", "token"]] = None,
    ):
        labels = post_process(labels)
        return CONVERT_FUNC_MAP[granularity](
            labels=labels,
            sample_outputs=sample_outputs,
            tokenizer=tokenizer,
            default_for_none=default_for_none,
            config=self.config,
        )

    def labels_to_metrics(self, labels: List[dict]):
        labels = post_process(labels)

        statements = [
            sum((
                list(sent_label.values())
                for sent_label in label.values()
            ), [])
            for label in labels
        ]  # extract statements in each sample

        metrics = {
            f"#{key}": [sum(map(lambda x: x[0] == key, s)) for s in statements]
            for key in self.config.verification_map.keys()
        }
        metrics.update({
            f"#Valid {key}": [sum(map(lambda x: x[0] == key and x[1] != '1', s)) for s in statements]
            for key in self.config.verification_map.keys()
        })
        metrics['#helpless'] = [sum(map(lambda x: x[1] == '1', s)) for s in statements]
        metrics["#statements"] = list(map(len, statements))
        metrics["accuracy"] = [
            div(metrics["#Correct"][i] + metrics["#Hedged correct"][i], metrics["#statements"][i])
            for i in range(len(labels))
        ]
        metrics["valid accuracy"] = [
            div(metrics["#Valid Correct"][i] + metrics["#Valid Hedged correct"][i], metrics["#statements"][i])
            for i in range(len(labels))
        ]
        metrics["ground"] = [
            1 - div(metrics["#Vague"][i], metrics["#statements"][i])
            for i in range(len(labels))
        ]
        metrics['info'] = [
            mean(list(map(lambda x: float(x[1]), s)))
            for s in statements
        ]
        metrics["info_reward"] = [
            sum(map(partial(label2info, config=self.config), label.values()))
            for label in labels
        ]  # sum up informativeness in every sentence to form an answer informativeness

        indices = [i for i, n in enumerate(metrics["#statements"]) if n != 0]
        all_metrics = {k: mean([v[i] for i in indices]) for k, v in metrics.items()}
        all_metrics['%refuse'] = 1 - len(indices) / len(metrics["#statements"])

        return metrics, all_metrics


def token_rewards_from_labels(
    sample_outputs: Union[Tensor, List[List[Tensor]]],
    labels: List[dict],
    tokenizer=None,
    default_for_none: float = 0,
    config=None,
):
    all_scores = []
    for output, label in tqdm(zip(sample_outputs, labels), total=len(sample_outputs), desc="Convert"):
        scores = [0. for _ in output]
        out = tokenizer.decode(output)
        token_to_char = [len(tokenizer.decode(output[: i + 1])) for i in range(len(output))]

        for sent, sent_label in label.items():
            sent2out_index = pylcs.lcs_string_idx(sent, out)
            sent_last_idx = np.array(sent2out_index).max()
            sent_token_last_idx = bisect.bisect_right(token_to_char, sent_last_idx)
            if sent_label:
                info_score = label2info(sent_label, config)
                scores[min(sent_token_last_idx, len(scores) - 1)] += info_score

                for stat, stat_label in sent_label.items():
                    stat = stat[:-1] if stat.endswith('.') else stat
                    sent = sent[:-1] if sent.endswith('.') else sent
                    stat2sent_index = lcs_sequence_idx(stat, sent)
                    for sent_idx in reversed(list(filter(lambda i: i != -1, stat2sent_index))):
                        out_idx = sent2out_index[sent_idx]
                        if out_idx != -1:
                            truth_score = label2truth(stat_label, config)
                            index = min(bisect.bisect_right(token_to_char, out_idx), len(scores) - 1)
                            scores[index] += truth_score
                            break
            else:
                scores[sent_token_last_idx] += default_for_none

        all_scores.append(scores)
    return all_scores


def sentence_rewards_from_labels(
    sample_outputs: Union[Tensor, List[List[Tensor]]],
    labels: List[dict],
    tokenizer=None,
    default_for_none: float = 0,
    config=None,
):
    all_scores = []
    for output, label in tqdm(zip(sample_outputs, labels), total=len(sample_outputs), desc="Convert"):
        scores = [0. for _ in output]
        out = tokenizer.decode(output, skip_special_tokens=True)
        token_to_char = [
            len(tokenizer.decode(output[: i + 1], skip_special_tokens=True))
            for i, t in enumerate(output)
            if t not in [tokenizer.pad_token_id, tokenizer.eos_token_id]
        ]

        for sent, sent_label in label.items():
            sent2out_index = pylcs.lcs_string_idx(sent, out)
            sent_last_idx = np.array(sent2out_index).max()
            sent_token_last_idx = bisect.bisect_right(token_to_char, sent_last_idx)

            if sent_label:
                truth_score = sum(label2truth(stat_label, config) for stat_label in sent_label.values())
                info_score = label2info(sent_label, config)
                scores[sent_token_last_idx] += truth_score + info_score
            else:
                scores[sent_token_last_idx] += default_for_none

        all_scores.append(scores)
    return all_scores


def paragraph_rewards_from_labels(
    sample_outputs: Union[Tensor, List[List[Tensor]]],
    labels: List[dict],
    tokenizer=None,
    default_for_none: float = 0,
    config=None,
):
    all_scores = []
    for output, label in tqdm(zip(sample_outputs, labels), total=len(sample_outputs), desc="Convert"):
        scores = [default_for_none for _ in output]
        if len(labels) > 0:
            info = sum(label2info(sent_label, config) for sent_label in label.values())
            truth = sum(sum(label2truth(stat_label, config) for stat_label in sent_label.values()) for sent_label in label.values())
            scores[-1] = info + truth
        all_scores.append(scores)
    return all_scores


def label2info(sent_label, config):
    return config.info_weight * log(max(
        config.epsilon, sum(map(lambda x: config.informative_map[x[1]], sent_label.values()))
    ) + config.mu)


def label2truth(stat_label, config):
    return config.truth_weight * config.verification_map[stat_label[0]] * abs(config.informative_map[stat_label[1]])


def post_process(labels: List[Dict]):
    for i, label in enumerate(labels):
        filtered = []
        for sent in label.keys():
            if "i'm sorry" in sent or \
                    "don't have any information" in sent or \
                    "do not have any information" in sent or \
                    "does not have any information" in sent or \
                    "does not have any information" in sent or \
                    "doesn't have any information" in sent or \
                    sent.endswith("?") or \
                    "may be able to provide more" in sent or \
                    "have any additional information" in sent or \
                    "need more context to" in sent or \
                    "requests for more context" in sent or \
                    "request for more context" in sent or \
                    "it is not possible to" in sent:
                filtered.append(sent)
        labels[i] = {sent: label[sent] for sent in label.keys() if sent not in filtered}
    return labels


CONVERT_FUNC_MAP = {
    "paragraph": paragraph_rewards_from_labels,
    "sentence": sentence_rewards_from_labels,
    "token": token_rewards_from_labels,
}
