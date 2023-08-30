import json
import logging
import queue
import re
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from dataclasses import dataclass
from json import JSONDecodeError
from typing import Any, Callable, Dict, List, Optional

from tqdm import tqdm

from source.feedback.completion import Completion, CompletionConfig
from source.feedback.retrieval import Retrieval, RetrieveConfig

logger = logging.getLogger(__name__)

EXTRACT_PROMPT = r'''- Find every sentence containing object facts.
- Break sentences into atomic statements.
- Skip the sentences without statements.
- If there is no valid sentence, output "No statements".
- Do not output any explanation or other words.
- Strictly follow the output format shown in the example.

Here is an example:
# Response
It is difficult to say which game has been released in more versions without more information, so I can only guess based on my training data.
Arthur's Magazine was likely started first. It was possibly founded in 1923 by Arthur K. Watson, a prominent publisher in the field of men's magazines.
First for Women, on the other hand, was not founded until 1989. It was created as a spin-off of Family Circle magazine, which was founded in 1957.

# Statements
>> Sentence 1: Arthur's Magazine was likely started first.
* Arthur's Magazine was likely started first.
>> Sentence 2: It was possibly founded in 1923 by Arthur K. Watson, a prominent publisher in the field of men's magazines.
* Arthur's Magazine was possibly founded in 1923.
* Arthur's Magazine was founded by Arthur K. Watson.
* Arthur K. Watson is a prominent publisher in the field of men's magazines.
>> Sentence 3: First for Women, on the other hand, was not founded until 1989.
* First for Women was not founded until 1989.
>> Sentence 4: It was created as a spin-off of Family Circle magazine, which was founded in 1957.
* First for Women was created as a spin-off of Family Circle magazine.
* Family Circle magazine was founded in 1957.

And then comes your task:
# Response
{response}

# Statements'''
VERIFY_PROMPT = '''Choose from "Correct", "Vague" and "Wrong" for the verification of the statement.
- "Correct": The statement is supported by the materials.
- "Vague":  Hard to determine the truthfulness of the statement based on the materials.
- "Wrong": The statement is negated by the materials.
Directly output the verification result without explanation.

Here is an example:
# Materials
- First for Women is a women's magazine published by Bauer Media Group in the USA. The magazine was started in 1989. It is based in Englewood Cliffs, New Jersey. In 2011 the circulation of the magazine was 1,310,696 copies."
- Arthur's Magazine (1844–1846) was an American literary periodical published in Philadelphia in the 19th century. Edited by T.S. Arthur, it featured work by Edgar A. Poe, J.H. Ingraham, Sarah Josepha Hale, Thomas G. Spear, and others. In May 1846 it was merged into "Godey's Lady's Book".
- The correct answer for the question "Which magazine was started first Arthur's Magazine or First for Women" may be "Arthur's Magazine".
# Statement
Arthur's Magazine was likely started first.
# Verification
Correct

And then comes your question:
# Materials
{materials}
# Statement
{statement}
# Verification'''
ASSESS_PROMPT = r'''Evaluate the helpfulness of the statement:
- "5": The statement answer the question.
- "4": The statement provides crucial information.
- "3": The statement contains relevant facts.
- "2": The statement is about other supplementary facts.
- "1": The statement is useless or not relevant at all.
Directly output the evaluation result without explanation.

Here is an example:
# Question
Which magazine was started first Arthur's Magazine founded by Arthur K. Watson or First for Women?
# Response
It is difficult to say which game has been released in more versions without more information, so I can only guess based on my training data.
Arthur's Magazine was likely started first. It was possibly founded in 1923 by Arthur K. Watson, a prominent publisher in the field of men's magazines.
First for Women, on the other hand, was not founded until 1989. It was created as a spin-off of Family Circle magazine, which was founded in 1957.
# Statement
Arthur's Magazine was possibly founded in 1923.
# Evaluation
4

And then comes your task:
# Question
{question}
# Response
{response}
# Statement
{statement}
# Evaluation'''


@dataclass
class AnnotatorConfig:
    retry_times: Optional[int] = 4
    sleep_time: Optional[int] = 1
    num_procs: Optional[int] = 64
    retrieval_config: RetrieveConfig = RetrieveConfig()
    completion_config: CompletionConfig = CompletionConfig()


class Annotator:
    def __init__(self, config: AnnotatorConfig):
        self.config = config
        self.completion = Completion(config.completion_config)
        self.retrieval = Retrieval(config.retrieval_config)
        self.executor = ThreadPoolExecutor(self.config.num_procs)

    def annotate(
        self,
        answers: List[str],
        questions: List[str],
        titles: List[List[str]],
        materials: Optional[List[List[str]]] = None,
        extra_materials: Optional[List[List[str]]] = None,
    ) -> List[Dict]:
        submit_queue, task_pool = queue.Queue(), []

        submit_bar = tqdm(desc='Submitted', total=len(answers))
        complete_bar = tqdm(desc='Completed', total=0)
        annotations = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))

        materials = materials or [[] for _ in range(len(answers))]
        extra_materials = extra_materials or [None for _ in range(len(answers))]
        for i, (answer, question, title, mater, extra) in enumerate(zip(
            answers, questions, titles, materials, extra_materials
        )):
            submit_queue.put({
                'id': i,
                'answer': answer,
                'question': question,
                'titles': title,
                'materials': mater,
                'extra_materials': extra,
                'task': 'extract',
            })

        func_mapping = {
            "extract": self.extract,
            "verify": self.verify,
            "assess": self.assess,
        }
        while True:
            if submit_queue.empty():
                if len(task_pool) != 0:
                    for future, task in task_pool:
                        if future.done():
                            task_pool.remove((future, task))
                            result = future.result()
                            complete_bar.update(1)

                            if task['task'] == 'extract':
                                submit_bar.update(1)
                                for sentence, statements in result.items():
                                    for statement in statements:
                                        task['sentence'] = sentence
                                        task['statement'] = statement

                                        task['task'] = 'verify'
                                        submit_queue.put(deepcopy(task))

                                        task['task'] = 'assess'
                                        submit_queue.put(deepcopy(task))
                            else:
                                annotations[task['id']][task['sentence']][task['statement']][task['task']] = result
                else:
                    break
            else:
                task = submit_queue.get()
                func = func_mapping[task['task']]
                future = self.executor.submit(func, **task)
                complete_bar.total += 1
                task_pool.append((future, task))

        annotations = [annotations[k] for k in range(len(answers))]
        for i, annotation in enumerate(annotations):
            for sentence, statements in annotation.items():
                for statement, result in statements.items():
                    annotations[i][sentence][statement] = [result['verify'], result['assess']]
        return annotations

    def extract(self, answer: str, *args, **kwargs):
        return self.label(
            build_extract_prompt(answer),
            extract_parse,
            temperature=1.,
            top_p=1.,
            max_tokens=4096
        )

    def verify(
        self,
        question: str,
        statement: str,
        titles: List[str],
        materials: List[str],
        extra_materials: Optional[List[str]] = None,
        *args, **kwargs
    ):
        return self.label(
            build_verify_prompt(
                question=question,
                statement=statement,
                titles=titles,
                materials=materials,
                extra_materials=extra_materials,
                retrieval=self.retrieval,
            ),
            verify_parse,
            temperature=0.3,
            top_p=0.3,
            max_tokens=32,
            default_for_failure='Vague'
        )

    def assess(self, question: str, statement: str, answer: str, *args, **kwargs):
        return self.label(
            build_assess_prompt(question, statement, answer),
            assess_parse,
            temperature=0.3,
            top_p=0.3,
            max_tokens=48,
            default_for_failure='1'
        )

    def label(self, message: str, parse_func: Callable, default_for_failure: Any = {}, **completion_kwargs):
        t = 0
        while t < self.config.retry_times:
            try:
                return parse_func(
                    self.completion(message, sleep_time=self.config.sleep_time, **completion_kwargs))
            except JSONDecodeError as e:
                logger.warning(f"ParseError: {e} with retry time {t=}.")
                t += 1
                if 'temperature' in completion_kwargs and completion_kwargs['temperature'] >= 1:
                    completion_kwargs['temperature'] += 1 / self.config.retry_times
        return default_for_failure


def build_extract_prompt(answer):
    return EXTRACT_PROMPT.replace("{response}", answer)


def build_verify_prompt(
    question: str,
    statement: str,
    titles: List[str],
    materials: List[str],
    retrieval: Retrieval,
    extra_materials: Optional[List[str]] = None,
):
    materials = materials or []
    if extra_materials is not None:
        titles = [''] * len(materials) + titles
        materials.extend(extra_materials)

    materials = retrieval(
        question=question,
        answer=statement,
        titles=titles,
        materials=materials,
    )
    materials = '\n'.join([f'- {met.strip()}' for met in materials])
    message = VERIFY_PROMPT
    for key, value in zip(["{materials}", "{statement}"], [materials, statement]):
        message = message.replace(key, value)
    return message


def build_assess_prompt(question: str, statement: str, answer: str):
    message = ASSESS_PROMPT
    for key, value in zip(["{question}", "{response}", "{statement}"], [question, answer, statement]):
        message = message.replace(key, value)
    return message


def extract_parse(completion: str):
    try:
        matched = re.findall(r"(((>> Sentence \d+:).+\n+(([\*-].+\n*)*)?)+)", completion)
        if len(matched) == 0:
            if "no" in completion.lower() and \
                    ("statement" in completion.lower() or "sentence" in completion.lower()):
                return {}
            else:
                assert False, "Extraction facing broken format"
        else:
            assert len(matched) == 1, "Extraction match more than one pattern"
            matched = list(filter(lambda l: len(l) > 0, matched[0][0].split('\n')))

            result = defaultdict(list)
            sentence = None
            for line in matched:
                # if line.startswith(">> S Sentence"):
                #     line = line.replace(">> S Sentence", ">> Sentence")
                if line.startswith('>> Sentence'):
                    sentence = re.split(r'>> Sentence \d+:', line)
                    sentence = [s.strip() for s in sentence if s.strip() != '']
                    assert len(sentence) == 1, f"{sentence} have more than one [>> Sentence \\d+:]"
                    sentence = sentence[0]
                elif line[0] in ['*', '-']:
                    assert sentence is not None, f"Facing None sentence but already get a statement"
                    result[sentence].append(line[1:].strip())
                else:
                    assert False, f"Unexpected {line=}"
            return result
    except AssertionError as e:
        raise JSONDecodeError(str(e), json.dumps(completion), 0) from e


def assess_parse(completion: str) -> str:
    try:
        for result in range(1, 6):
            if str(result) in completion:
                return str(result)
        assert False, f"Assess return \"{completion}\" which is not valid"
    except AssertionError as e:
        raise JSONDecodeError(str(e), json.dumps(completion), 0) from e


def verify_parse(completion: str) -> str:
    try:
        for result in ['Correct', 'Vague', 'Wrong']:
            if result.lower() in completion.lower():
                return result
        if ("need more context" in completion or
                "need access to" in completion or
                "provide more context" in completion or
            "need additional context" in completion or
                "need additional information" in completion or
                "not supported" in completion):
            return "Vague"
        assert False, f"Verify return \"{completion}\" which is not valid"
    except AssertionError as e:
        raise JSONDecodeError(str(e), json.dumps(completion), 0) from e
