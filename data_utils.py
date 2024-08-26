from torch.utils.data import Dataset
import os
import json
import torch
from transformers import BartTokenizer, PegasusTokenizer
from compare_mt.rouge.rouge_scorer import RougeScorer
from nltk import sent_tokenize, word_tokenize


def to_cuda(batch, gpuid):
    for n in batch:
        if n != "data" and n != "gold" and n != "source":
            batch[n] = batch[n].to(gpuid)


class FactSumGANDataset(Dataset):
    def __init__(self, fdir, model_type, max_len=-1, is_test=False, total_len=512, is_sorted=True, max_num=-1,
                 is_untok=True, is_pegasus=False, num=-1, ranking_metric=None):
        """ 数据格式: 原文, 参考摘要, [(候选摘要_i, 事实性分数_i)] """
        self.isdir = os.path.isdir(fdir)
        if self.isdir:
            self.fdir = fdir
            if num > 0:
                self.num = min(len(os.listdir(fdir)), num)
            else:
                self.num = len(os.listdir(fdir))
        else:
            with open(fdir) as f:
                self.files = [x.strip() for x in f]
            if num > 0:
                self.num = min(len(self.files), num)
            else:
                self.num = len(self.files)
        if is_pegasus:
            self.tok = PegasusTokenizer.from_pretrained(model_type, verbose=False)
        else:
            self.tok = BartTokenizer.from_pretrained(model_type, verbose=False)
        self.maxlen = max_len
        self.is_test = is_test
        self.total_len = total_len
        self.sorted = is_sorted
        self.maxnum = max_num
        self.is_untok = is_untok
        self.is_pegasus = is_pegasus
        self.ranking_metric = ranking_metric

    def __len__(self):
        return self.num

    def __getitem__(self, idx):
        def process(x):
            return sent_tokenize(" ".join(word_tokenize(x.strip())))

        rouge_scorer = RougeScorer(['rouge1', 'rouge2', 'rougeLsum'], use_stemmer=True)
        if self.isdir:
            with open(os.path.join(self.fdir, "%d.json" % idx), "r") as f:
                data = json.load(f)
        else:
            with open(self.files[idx]) as f:
                data = json.load(f)
        if self.is_untok:
            article = data["article_untok"]
        else:
            article = data["article"]
        src_txt = " ".join(article)
        src = self.tok.batch_encode_plus([src_txt], max_length=self.total_len, return_tensors="pt",
                                         pad_to_max_length=False, truncation=True)
        src_pad_max = self.tok.batch_encode_plus([" ".join(data["article_untok"])], max_length=self.total_len,
                                                 return_tensors="pt", pad_to_max_length=True, truncation=True)
        src_input_ids = src["input_ids"].squeeze(0)
        src_pad_max_input_ids = src_pad_max["input_ids"].squeeze(0)
        src_pad_max_attmasks = src_pad_max["attention_mask"].squeeze(0)
        if self.is_untok:
            abstract = data["abstract_untok"]
        else:
            abstract = data["abstract"]
        if self.maxnum > 0:
            candidates = data["candidates_untok"][:self.maxnum]
            _candidates = data["candidates"][:self.maxnum]
            data["candidates"] = _candidates
        for index, cand in enumerate(candidates):
            rouge_score = rouge_scorer.score("\n".join(process(" ".join(data["abstract_untok"]))),
                                             "\n".join(process(" ".join(cand[0]))))
            rouglsum = rouge_score["rougeLsum"].fmeasure
            cand[1].append(rouglsum)
            _candidates[index][1].append(rouglsum)

        if self.sorted:
            if self.ranking_metric == "factCC":
                candidates = sorted(candidates, key=lambda x: x[1][0], reverse=True)
                _candidates = sorted(_candidates, key=lambda x: x[1][0], reverse=True)
            elif self.ranking_metric == "factCC_and_QAGS":
                candidates = sorted(candidates, key=lambda x: (0.65 * x[1][0] + 0.35 * x[1][1]), reverse=True)
                _candidates = sorted(_candidates, key=lambda x: (0.65 * x[1][0] + 0.35 * x[1][1]), reverse=True)
            elif self.ranking_metric == "factCC_and_Rouge":
                candidates = sorted(candidates, key=lambda x: (0.7 * x[1][0] + 0.3 * x[1][2]), reverse=True)
                _candidates = sorted(_candidates, key=lambda x: (0.7 * x[1][0] + 0.3 * x[1][2]), reverse=True)
            else:
                candidates = sorted(candidates, key=lambda x: (0.7 * x[1][1] + 0.3 * x[1][2]), reverse=True)
                _candidates = sorted(_candidates, key=lambda x: (0.7 * x[1][1] + 0.3 * x[1][2]), reverse=True)
            data["candidates"] = _candidates
        if not self.is_untok:
            candidates = _candidates
        gold = " ".join(data["abstract_untok"])
        cand_txt = [" ".join(abstract)] + [" ".join(x[0]) for x in candidates]
        cand = self.tok.batch_encode_plus(cand_txt, max_length=self.maxlen, return_tensors="pt",
                                          pad_to_max_length=False, truncation=True, padding=True)
        candidate_ids = cand["input_ids"]
        if self.is_pegasus:
            # 添加起始符
            _candidate_ids = candidate_ids.new_zeros(candidate_ids.size(0), candidate_ids.size(1) + 1)
            _candidate_ids[:, 1:] = candidate_ids.clone()
            _candidate_ids[:, 0] = self.tok.pad_token_id
            candidate_ids = _candidate_ids
        result = {
            "src_input_ids": src_input_ids,
            "candidate_ids": candidate_ids,
            "gold": gold,
            "source": src_txt,
            "src_pad_max_input_ids": src_pad_max_input_ids,
            "src_pad_max_attmasks": src_pad_max_attmasks
        }
        if self.is_test:
            result["data"] = data
        return result


def collate_mp_fact_sum_gan(batch, pad_token_id, is_test=False):
    def pad(sample, max_length=-1):
        if max_length < 0:
            max_length = max(x.size(0) for x in sample)
        result_tensor = torch.ones(len(sample), max_length, dtype=sample[0].dtype) * pad_token_id
        for (i, x) in enumerate(sample):
            result_tensor[i, :x.size(0)] = x
        return result_tensor

    src_input_ids = pad([x["src_input_ids"] for x in batch])
    candidate_ids = [x["candidate_ids"] for x in batch]
    max_len = max([max([len(c) for c in x]) for x in candidate_ids])
    candidate_ids = [pad(x, max_len) for x in candidate_ids]
    candidate_ids = torch.stack(candidate_ids)  # [bz,cand_num,max_len]
    src_pad_max_input_ids = [x["src_pad_max_input_ids"] for x in batch]
    src_pad_max_attmasks = [x["src_pad_max_attmasks"] for x in batch]
    src_pad_max_input_ids = torch.stack(src_pad_max_input_ids, dim=0)
    src_pad_max_attmasks = torch.stack(src_pad_max_attmasks, dim=0)
    gold = [x["gold"] for x in batch]
    source = [x["source"] for x in batch]
    result = {
        "src_input_ids": src_input_ids,  # [bz,max_len_of_src]
        "candidate_ids": candidate_ids,  # [bz,cand_num,max_len_of_cand]
        "gold": gold,  # 该batch的全部参考摘要文本
        "source": source,
        "src_pad_max_input_ids": src_pad_max_input_ids,  # [bz,total_len]
        "src_pad_max_attmasks": src_pad_max_attmasks  # [bz,total_len]
    }
    if is_test:
        data = [x["data"] for x in batch]
        result["data"] = data

    return result
