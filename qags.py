import logging
import os

from typing import Dict, List, Union

import pysbd

from module_entity import load_ner
from module_question import load_qa, load_qg
from qa_utils import Config, qags_score

os.environ["TOKENIZERS_PARALLELISM"] = "false"

logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("flair").setLevel(logging.ERROR)


class FactSumm:

    def __init__(
            self,
    ):
        """
        FactSumm object used to calculate Factual Consistency score of Abstractive Summarization model

        Args:
            ner_model (str, optional): NER model to be used (Flair or HuggingFace). Defaults to None.
            qg_model (str, optional): QA model to be used (HuggingFace). Defaults to None.
            qa_model (str, optional): QG model to be used (HuggingFace). Defaults to None.
        """
        self.config = Config()
        self.segmenter = pysbd.Segmenter(language="en", clean=False)

        # NER,QG, QA models supported by HuggingFace can be used (default can be found in `config.py`)
        self.ner = self.config.NER_MODEL
        self.qg = self.config.QG_MODEL
        self.qa = self.config.QA_MODEL

    def _segment(self, text: str) -> List[str]:
        """
        Segment input text into (possibly) multiple sentences

        Args:
            text (str): text to be segmented

        Returns:
            List[str]: list of segmented lines

        """
        return [line.strip() for line in self.segmenter.segment(text)]

    def extract_qas(
            self,
            source: str,
            summary: str,
            device: str,
    ) -> float:
        """
        Extract Question & Answering Pair generated from Question Generation module

            See also https://arxiv.org/abs/2004.04228

        Args:
            source (str): original source
            summary (str): generated summary
            device (str): device info

        """
        if isinstance(self.qg, str) and isinstance(self.qa, str):
            self.qg = load_qg(self.qg, device)
            self.qa = load_qa(self.qa, device)

        if isinstance(self.ner, str):
            self.ner = load_ner(self.ner, device)

        summary_lines = self._segment(summary)
        summary_ents = self.ner(summary_lines)

        summary_qas = self.qg(summary_lines, summary_ents)

        source_answers = self.qa(source, summary_qas)
        summary_answers = self.qa(summary, summary_qas)

        qa_score = qags_score(source_answers, summary_answers)

        return qa_score

    def __call__(
            self,
            sources: Union[List[str], str],
            summaries: Union[List[str], str],
            device: str,
    ) -> Dict:
        if isinstance(sources, str) and isinstance(summaries, str):
            sources = [sources]
            summaries = [summaries]

        if len(sources) != len(summaries):
            # yapf:disable
            raise ValueError("`sources` and `summaries` must have the same number of elements!")
            # yapf:enable

        num_pairs = len(sources)
        qags_scores = 0

        for source, summary in zip(sources, summaries):
            _qags_score = self.extract_qas(
                source,
                summary,
                device
            )
            qags_scores += _qags_score

        return {
            "qa_score": qags_scores / num_pairs,
        }
