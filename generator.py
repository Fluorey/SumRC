import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from transformers import BartTokenizer, PegasusTokenizer
from modeling_bart import BartScorer
from modeling_pegasus import PegasusScorer
from typing import Callable, Iterable, List, Optional


def RankingLoss(score, summary_score=None, margin=0, gold_margin=0, gold_weight=1, no_gold=False, no_cand=False):
    ones = torch.ones_like(score)
    loss_func = torch.nn.MarginRankingLoss(0.0)
    TotalLoss = loss_func(score, score, ones)
    # candidate loss
    n = score.size(1)
    if not no_cand:
        for i in range(1, n):
            pos_score = score[:, :-i]
            neg_score = score[:, i:]
            pos_score = pos_score.contiguous().view(-1)
            neg_score = neg_score.contiguous().view(-1)
            ones = torch.ones_like(pos_score)
            loss_func = torch.nn.MarginRankingLoss(margin * i)
            loss = loss_func(pos_score, neg_score, ones)
            TotalLoss += loss
    if no_gold:
        return TotalLoss
    # gold summary loss
    pos_score = summary_score.unsqueeze(-1).expand_as(score)
    neg_score = score
    pos_score = pos_score.contiguous().view(-1)
    neg_score = neg_score.contiguous().view(-1)
    ones = torch.ones_like(pos_score)
    loss_func = torch.nn.MarginRankingLoss(gold_margin)
    TotalLoss += gold_weight * loss_func(pos_score, neg_score, ones)
    return TotalLoss


class Generator(nn.Module):

    def __init__(self, model_path, pad_token_id, is_pegasus):
        super(Generator, self).__init__()
        self.is_pegasus = is_pegasus  # 表示是否用基于Pegasus的预训练模型
        self.pad_token_id = pad_token_id  # 填充符的序号
        if is_pegasus:
            self.model = PegasusScorer.from_pretrained(model_path)
        else:
            self.model = BartScorer.from_pretrained(model_path)

    def forward(self, input_ids, candidate_ids, normalize=True, score_mode="log", length_penalty=1, require_gold=True,
                adding=0):
        """
        前向传播
        candidate_ids:[bz,cand_num,seq_len]
        """
        batch_size = input_ids.size(0)
        """计算attention_mask"""
        input_mask = input_ids != self.pad_token_id
        cand_mask = candidate_ids != self.pad_token_id
        cand_mask[:, :, 0] = 1

        output = self.model(
            input_ids=input_ids,
            attention_mask=input_mask,
            decoder_input_ids=candidate_ids,
            decoder_attention_mask=cand_mask,
            output_hidden_states=True
        )
        output = output[0]  # [bz x cand_num, seq_len, word_dim]
        output = output.view(batch_size, -1, output.size(1), output.size(2))  # [bz, cand_num, seq_len, word_dim]
        probs = output[:, 0]  # [bz, seq_len, word_dim]
        output = output[:, :, :-1]  # 截掉最后一个token
        candidate_ids = candidate_ids[:, :, 1:]  # 右移
        cand_mask = candidate_ids != self.pad_token_id  # [bz, cand_num, seq_len]
        cand_mask = cand_mask.float()
        # TODO:这里直接用.float()行吗？
        candidate_ids = candidate_ids.unsqueeze(-1)  # [bz, cand_num, seq_len, 1]
        """计算概率"""
        if normalize:
            if score_mode == "log":
                _output = F.log_softmax(output, dim=3)
            else:
                _output = F.softmax(output, dim=3)
            scores = torch.gather(_output, 3, candidate_ids).squeeze(-1)  # [bz, cand_num, seq_len]
            score_dis = torch.mul(scores, cand_mask)  # [bz, cand_num, seq_len]
        else:
            scores = torch.gather(output, 3, candidate_ids).squeeze(-1)  # [bz, cand_num, seq_len]
            score_dis = torch.mul(scores, cand_mask)  # [bz, cand_num, seq_len]
        scores = score_dis.sum(-1) / ((cand_mask.sum(-1) + adding) ** length_penalty)  # [bz, cand_num]

        """score为生成器对候选摘要赋予的概率"""
        if require_gold:
            output = {'score': scores[:, 1:], "summary_score": scores[:, 0], "score_dis": score_dis, "probs": probs}
        else:
            output = {'score': scores, "score_dis": score_dis, "probs": probs}
        return output

    def batchPG(self, input_ids, target_ids, reward):
        """
        计算生成器强化学习下的策略梯度
        input_ids:[bz,seq_len]
        target_ids:[bz,seq_len]，代表模型生成的摘要
        reward:[bz]
        """

        batch_size, seq_len = input_ids.size()
        target_ids = target_ids.unsqueeze(1)  # [bz,1,seq_len]
        out = self.forward(input_ids, target_ids)
        score_dis = out['score_dis'].squeeze(1)  # [bz,seq_len]

        pg = 0
        for i in range(batch_size):
            score_sum = score_dis[i].sum(-1)
            score_sum = torch.exp(score_sum * 0.01)
            pg += score_sum * reward[i]  # P(y_t|Y_1:Y_{t-1}) * Q

        return pg / batch_size

    def generate(
            self,
            input_ids: Optional[torch.LongTensor] = None,
            max_length: Optional[int] = None,
            min_length: Optional[int] = None,
            do_sample: Optional[bool] = None,
            early_stopping: Optional[bool] = None,
            num_beams: Optional[int] = None,
            temperature: Optional[float] = None,
            top_k: Optional[int] = None,
            top_p: Optional[float] = None,
            repetition_penalty: Optional[float] = None,
            bad_words_ids: Optional[Iterable[int]] = None,
            bos_token_id: Optional[int] = None,
            pad_token_id: Optional[int] = None,
            eos_token_id: Optional[int] = None,
            length_penalty: Optional[float] = None,
            no_repeat_ngram_size: Optional[int] = None,
            encoder_no_repeat_ngram_size: Optional[int] = None,
            num_return_sequences: Optional[int] = None,
            max_time: Optional[float] = None,
            decoder_start_token_id: Optional[int] = None,
            use_cache: Optional[bool] = None,
            num_beam_groups: Optional[int] = None,
            diversity_penalty: Optional[float] = None,
            prefix_allowed_tokens_fn: Optional[Callable[[int, torch.Tensor], List[int]]] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            output_scores: Optional[bool] = None,
            return_dict_in_generate: Optional[bool] = None,
            forced_bos_token_id: Optional[int] = None,
            forced_eos_token_id: Optional[int] = None,
            remove_invalid_values: Optional[bool] = None,
            synced_gpus: Optional[bool] = None,
            **model_kwargs,
    ):

        return self.model.generate(input_ids=input_ids,
                                   max_length=max_length,
                                   min_length=min_length,
                                   do_sample=do_sample,
                                   early_stopping=early_stopping,
                                   num_beams=num_beams,
                                   temperature=temperature,
                                   top_k=top_k,
                                   top_p=top_p,
                                   repetition_penalty=repetition_penalty,
                                   bad_words_ids=bad_words_ids,
                                   bos_token_id=bos_token_id,
                                   pad_token_id=pad_token_id,
                                   eos_token_id=eos_token_id,
                                   length_penalty=length_penalty,
                                   no_repeat_ngram_size=no_repeat_ngram_size,
                                   encoder_no_repeat_ngram_size=encoder_no_repeat_ngram_size,
                                   num_return_sequences=num_return_sequences,
                                   max_time=max_time,
                                   decoder_start_token_id=decoder_start_token_id,
                                   use_cache=use_cache,
                                   num_beam_groups=num_beam_groups,
                                   diversity_penalty=diversity_penalty,
                                   prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
                                   output_attentions=output_attentions,
                                   output_hidden_states=output_hidden_states,
                                   output_scores=output_scores,
                                   return_dict_in_generate=return_dict_in_generate,
                                   forced_bos_token_id=forced_bos_token_id,
                                   forced_eos_token_id=forced_eos_token_id,
                                   remove_invalid_values=remove_invalid_values,
                                   synced_gpus=synced_gpus,
                                   **model_kwargs)

    def scoring_mode(self):
        self.model.model.scoring_mode()

    def generation_mode(self):
        self.model.model.generation_mode()
