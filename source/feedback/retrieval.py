from typing import List, Optional

import nltk
import numpy as np
from dataclasses import dataclass
from nltk import sent_tokenize
from nltk.corpus import stopwords

try:
    STOPS = stopwords.words('english') + ['!', ',', '.', '?', '-s', '-ly', '</s> ', 's']
except LookupError:
    nltk.download('stopwords')
try:
    _ = sent_tokenize("test test")
except LookupError:
    nltk.download('punkt')


def word_tokenize(text: str, use_stops: bool = True) -> List[str]:
    tokenized = nltk.word_tokenize(text)
    if use_stops:
        tokenized = list(filter(lambda w: w not in STOPS, tokenized))
    return tokenized


def keep_topk(arr: np.ndarray, k: int = 2) -> np.ndarray:
    max_indices = np.argsort(-arr, axis=None)
    if len(max_indices) > k:
        arr[arr < arr[max_indices[k - 1]]] = 0
    return arr


@dataclass
class RetrieveConfig:
    batch_size: int = -1
    retrieval_type: str = 'bm25'
    max_words_per_sec: Optional[int] = 64
    max_secs: Optional[int] = 2


class Retrieval:
    def __init__(self, config: RetrieveConfig):
        self.config = config
        self.retrieval_type = config.retrieval_type
        self.batch_size = config.batch_size
        assert config.retrieval_type == "bm25" or config.retrieval_type.startswith("gtr-")

        self.encoder = None

    def load_encoder(self):
        from sentence_transformers import SentenceTransformer
        encoder = SentenceTransformer("sentence-transformers/" + self.retrieval_type)
        encoder = encoder.cuda()
        encoder = encoder.eval()
        self.encoder = encoder
        assert self.batch_size is not None

    def build_sections(self, titles: List[str], materials: List[str]):
        sections, tokenized_secs = [], []
        for title, material in zip(titles, materials):
            section, tokenized_sec = '', []
            for sent in sent_tokenize(material):
                tokenized_sec.extend(word_tokenize(sent, use_stops=False))
                section += f' {sent}'

                if len(tokenized_sec) > self.config.max_words_per_sec:
                    if title != '':
                        tokenized_sec = word_tokenize(title) + tokenized_sec
                        section = f"<{title}>{section}"

                    tokenized_secs.append(tokenized_sec)
                    sections.append(section.strip())
                    section, tokenized_sec = '', []

            if section != '' or tokenized_secs != []:
                tokenized_secs.append(tokenized_sec)
                sections.append(section.strip())

        return sections, tokenized_secs

    def get_bm25_passages(self, answer: str, titles: List[str], materials: List[str], question: str = None):
        from rank_bm25 import BM25Okapi
        sections, tokenized_secs = self.build_sections(titles, materials)
        bm25 = BM25Okapi(tokenized_secs)

        queries = sent_tokenize(answer)
        if len(queries) == 0:
            if question is not None:
                queries.append(question)
            else:
                return sections[:self.config.max_secs]

        scores = keep_topk(bm25.get_scores(word_tokenize(queries[0])), k=self.config.max_secs)
        for sec in queries[1:]:
            scores += keep_topk(bm25.get_scores(word_tokenize(sec)), k=self.config.max_secs)
        indices = np.argsort(-scores)[:self.config.max_secs]
        return [sections[i] for i in indices if scores[i] != 0]

    def get_gtr_passages(self, question: str, answer: str, titles: List[str], materials: List[str]):
        if self.encoder is None:
            self.load_encoder()
        sections = self.build_sections(titles, materials)[0]
        passage_vectors = self.encoder.encode(sections, batch_size=self.batch_size, device=self.encoder.device)

        queries = sent_tokenize(answer)
        if question is not None:
            queries.append(question)

        query_vectors = self.encoder.encode(
            queries,
            batch_size=self.batch_size,
            device=self.encoder.device
        )[0]
        scores = np.inner(query_vectors, passage_vectors)
        indices = np.argsort(-scores)[:self.config.max_secs]
        return [sections[i] for i in indices]

    def __call__(self, answer: str, titles: List[str], materials: List[str], question: str = None):
        if self.retrieval_type == "bm25":
            return self.get_bm25_passages(question=question, answer=answer, titles=titles, materials=materials)
        else:
            return self.get_gtr_passages(question=question, answer=answer, titles=titles, materials=materials)
