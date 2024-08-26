from typing import List

from flair.data import Sentence
from flair.models import SequenceTagger
from rich import print


def load_ner(model: str, device: str) -> object:
    """
    Load Named Entity Recognition model from HuggingFace hub

    Args:
        model (str): model name to be loaded
        device (str): device info

    Returns:
        object: Pipeline-based Named Entity Recognition model

    """
    try:
        ner = SequenceTagger.load(model).to(device)
    except UnboundLocalError:
        print("Input model is not supported by Flair")

    def extract_entities_flair(sentences: List[str]):
        result = list()

        for sentence in sentences:
            sentence = Sentence(sentence)
            ner.predict(sentence)
            line_result = sentence.to_dict(tag_type="ner")

            cache = dict()
            dedup = list()

            for entity in line_result["entities"]:
                existence = cache.get(entity["text"], False)

                if not existence:
                    dedup.append({
                        "word": entity["text"],
                        "entity": entity["labels"][0]['value'],
                        "start": entity["start_pos"],
                        "end": entity["end_pos"],
                    })
                    cache[entity["text"]] = True

            result.append(dedup)

        return result

    return extract_entities_flair