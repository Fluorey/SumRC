import json
import torch
import os
from tqdm import tqdm
from nltk import sent_tokenize
from transformers import BertForSequenceClassification, BertTokenizer
from qags import FactSumm
import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="transformers.pipelines.base")

gpuid = 0
device = f"cuda:{gpuid}"
factCC_path = "huggingface/factCC2"
factCC_model = BertForSequenceClassification.from_pretrained(factCC_path).to(device)
factCC_tok = BertTokenizer.from_pretrained(factCC_path)
factsumm = FactSumm()


def collect_diverse_beam_data(cand_num, src_dir, tgt_dir, split, lower):
    tgt_dir = os.path.join(tgt_dir, split)
    cands = []
    cands_untok = []
    cnt = 0
    with open(os.path.join(src_dir, f"{split}.source.tokenized")) as src, open(
            os.path.join(src_dir, f"{split}.target.tokenized")) as tgt, open(
        os.path.join(src_dir, f"{split}.source")) as src_untok, open(
        os.path.join(src_dir, f"{split}.target")) as tgt_untok:
        with open(os.path.join(src_dir, f"{split}.out.tokenized")) as f_1, open(
                os.path.join(src_dir, f"{split}.out")) as f_2:
            for (x, y) in zip(f_1, f_2):
                x = x.strip()
                if lower:
                    x = x.lower()
                cands.append(x)
                y = y.strip()
                if lower:
                    y = y.lower()
                cands_untok.append(y)
                if len(cands) == cand_num:
                    src_line = src.readline()
                    src_line = src_line.strip()
                    if lower:
                        src_line = src_line.lower()
                    tgt_line = tgt.readline()
                    tgt_line = tgt_line.strip()
                    if lower:
                        tgt_line = tgt_line.lower()
                    src_line_untok = src_untok.readline()
                    src_line_untok = src_line_untok.strip()
                    if lower:
                        src_line_untok = src_line_untok.lower()
                    tgt_line_untok = tgt_untok.readline()
                    tgt_line_untok = tgt_line_untok.strip()
                    if lower:
                        tgt_line_untok = tgt_line_untok.lower()
                    yield (src_line, tgt_line, cands, src_line_untok, tgt_line_untok, cands_untok,
                           os.path.join(tgt_dir, f"{cnt}.json"))
                    cands = []
                    cands_untok = []
                    cnt += 1


def build_diverse_beam(input):
    src_line, tgt_line, cands, src_line_untok, tgt_line_untok, cands_untok, tgt_dir = input
    cands = [sent_tokenize(x) for x in cands]
    abstract = sent_tokenize(tgt_line)
    article = sent_tokenize(src_line)

    def compute_factuality(source, hyp):
        input_dict = factCC_tok(" ".join(source), " ".join(hyp), max_length=512, padding='max_length',
                                truncation='only_first',
                                return_tensors='pt')
        input_dict = input_dict.to(device)
        logits = torch.softmax(factCC_model(**input_dict).logits, dim=1)
        factuality_score = logits[0][0].item()
        qags_score = factsumm.extract_qas(" ".join(source), " ".join(hyp), f"cuda:0")
        return [factuality_score, qags_score]

    candidates = [(x, compute_factuality(article, x)) for x in cands]
    cands_untok = [sent_tokenize(x) for x in cands_untok]
    abstract_untok = sent_tokenize(tgt_line_untok)
    article_untok = sent_tokenize(src_line_untok)
    candidates_untok = [(cands_untok[i], candidates[i][1]) for i in range(len(candidates))]
    output = {
        "article": article,
        "abstract": abstract,
        "candidates": candidates,
        "article_untok": article_untok,
        "abstract_untok": abstract_untok,
        "candidates_untok": candidates_untok,
    }
    with open(tgt_dir, "w") as f:
        json.dump(output, f)


def make_diverse_beam_data(cand_num, src_dir, tgt_dir, split, lower):
    with open(os.path.join(src_dir, f"{split}.source")) as f:
        num = sum(1 for _ in f)
    data = collect_diverse_beam_data(cand_num, src_dir, tgt_dir, split, lower)
    ctx = torch.multiprocessing.get_context("spawn")
    with ctx.Pool(6) as pool:
        for _ in tqdm(pool.imap_unordered(build_diverse_beam, data, chunksize=64), total=num, ascii=True, desc=src_dir):
            pass
    print("finish")


if __name__ == "__main__":
    is_cnndm = False
    cand_num = 6
    splits = ["train", "test", "val"]
    lower = True

    if is_cnndm:
        src_dir_original = "examples/raw_data/cnn_dm"
        tgt_dir = "examples/processed_data/cnn_dm"

    else:
        src_dir_original = "examples/raw_data/xsum"
        tgt_dir = "examples/processed_data/xsum"

    for split in splits:
        src_dir = os.path.join(src_dir_original, split)
        make_diverse_beam_data(cand_num, src_dir, tgt_dir, split, lower)
