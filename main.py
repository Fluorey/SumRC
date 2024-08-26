import torch
import torch.nn as nn
import math
import torch.optim as optim
import argparse
import numpy as np
import os
import random
from transformers import BartTokenizer, PegasusTokenizer, BartForConditionalGeneration, PegasusForConditionalGeneration
from transformers import BertForSequenceClassification, BertTokenizer
from utils import Recorder
from data_utils import to_cuda, collate_mp_fact_sum_gan, FactSumGANDataset
from torch.utils.data import DataLoader
import torch.distributed as dist
import torch.multiprocessing as mp
from functools import partial
from generator import Generator, RankingLoss
import logging
from compare_mt.rouge.rouge_scorer import RougeScorer
from qags import FactSumm

from nltk import sent_tokenize, word_tokenize
from config import cnndm_setting, xsum_setting
from tqdm import tqdm

import warnings

warnings.filterwarnings("ignore")

"""一些相关参数"""
cuda = True
gpuids = [0]
model_pt = ""
discriminator_pt = ""
config = "xsum"
evaluate = True
do_reranking = False
do_generation = True

log = True
port = 12355


def to_evaluation(args):
    # 加载数据
    if config == "cnndm":
        cnndm_setting(args)
    else:
        xsum_setting(args)
    """------------------------------------------------------------------------------------------------"""
    # if args.is_pegasus:
    #     tok = PegasusTokenizer.from_pretrained(args.model_type)
    # else:
    #     tok = BartTokenizer.from_pretrained(args.model_type)

    device = f'cuda:{gpuids[0]}'

    # 构造模型
    """------------------------------------------------------------------------------------------------"""
    # model_path = args.pretrained if args.pretrained is not None else args.model_type
    # model = Generator(model_path, tok.pad_token_id, args.is_pegasus)
    # if cuda:
    #     model = model.to(f"cuda:{gpuids[0]}")
    """------------------------------------------------------------------------------------------------"""
    model_name = "fact_FC_6647_r1_2715_r2_999_rl_2301"
    eval_model_variant = "factCC_num16"
    # model.load_state_dict(
    #     torch.load(os.path.join(f"cache/{eval_model_variant}", f"{model_name}.bin"), map_location=f'cuda:{gpuids[0]}'))
    # model.eval()

    """------------------------------------------------------------------------------------------------"""

    def mkdir(path):
        if not os.path.exists(path):
            os.mkdir(path)

    print(model_name)
    root_dir = f"result/{eval_model_variant}/{model_name}"
    # mkdir(root_dir)
    rouge_scorer = RougeScorer(['rouge1', 'rouge2', 'rougeLsum'], use_stemmer=True)

    if do_generation:
        rouge1, rouge2, rougeLsum, factcc, qags = 0, 0, 0, 0, 0
        # tokenizer = tok
        # count = 1
        # bsz = 8
        """用模型生成摘要"""
        """------------------------------------------------------------------------------------------------"""
        total_num = len(os.listdir(f"examples/processed_data/{args.dataset}/test"))
        # model.generation_mode()
        # with open(f'examples/processed_data/{args.dataset}/test.source') as source, open(
        #         os.path.join(root_dir, "test.out"), 'w') as fout:
        #     sline = source.readline().strip()
        #     slines = [sline]
        #     for sline in tqdm(source, total=total_num, ascii=True, desc=f"用模型{model_name}输出摘要"):
        #         if count % bsz == 0:
        #             with torch.no_grad():
        #                 dct = tokenizer.batch_encode_plus(slines, max_length=args.total_len, return_tensors="pt",
        #                                                   pad_to_max_length=True, truncation=True)
        #                 summaries = model.model.generate(
        #                     input_ids=dct["input_ids"].to(device),
        #                     attention_mask=dct["attention_mask"].to(device),
        #                     max_length=args.gen_max_len + 2,
        #                     # +2 from original because we start at step=1 and stop before max_length
        #                     min_length=args.gen_min_len + 1,  # +1 from original because we start at step=1
        #                     no_repeat_ngram_size=3,
        #                     num_beams=args.num_beams,
        #                     length_penalty=args.length_penalty,
        #                     early_stopping=True,
        #                 )
        #                 dec = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=False) for g
        #                        in summaries]
        #             for hypothesis in dec:
        #                 hypothesis = hypothesis.replace("\n", " ")
        #                 fout.write(hypothesis + '\n')
        #                 fout.flush()
        #             slines = []
        #         sline = sline.strip()
        #         if len(sline) == 0:
        #             sline = " "
        #         slines.append(sline)
        #         count += 1
        #     if slines != []:
        #         with torch.no_grad():
        #             dct = tokenizer.batch_encode_plus(slines, max_length=args.total_len, return_tensors="pt",
        #                                               pad_to_max_length=True, truncation=True)
        #             summaries = model.model.generate(
        #                 input_ids=dct["input_ids"].to(device),
        #                 attention_mask=dct["attention_mask"].to(device),
        #                 max_length=args.gen_max_len + 2,
        #                 # +2 from original because we start at step=1 and stop before max_length
        #                 min_length=args.gen_min_len + 1,  # +1 from original because we start at step=1
        #                 no_repeat_ngram_size=3,
        #                 num_beams=args.num_beams,
        #                 length_penalty=args.length_penalty,
        #                 early_stopping=True,
        #             )
        #             dec = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=False) for g in
        #                    summaries.cpu()]
        #             for hypothesis in dec:
        #                 hypothesis = hypothesis.replace("\n", " ")
        #                 fout.write(hypothesis + '\n')
        #                 fout.flush()
        # """清除一部分变量释放内存"""
        # del model, summaries

        factcc_model = BertForSequenceClassification.from_pretrained("huggingface/factCC2").to(device)
        factcc_tok = BertTokenizer.from_pretrained("huggingface/factCC2")
        factsumm = FactSumm()

        # 对生成的摘要计算指标值
        def process(x):
            return sent_tokenize(" ".join(word_tokenize(x.strip())))

        with open(os.path.join(root_dir, "test.out")) as fout, open(
                f'examples/processed_data/{args.dataset}/test.target') as target, open(
            f'examples/processed_data/{args.dataset}/test.source') as source:
            for (hyp, ref, src) in tqdm(zip(fout, target, source), total=total_num, ascii=True,
                                        desc="对输出摘要进行评估", colour="#99FFFF"):
                hyp = hyp.strip()
                _hypothesis = hyp
                ref = ref.strip()
                hyp = process(hyp)
                ref = process(ref)
                source_article = src.strip()
                # 计算Rouge
                score = rouge_scorer.score("\n".join(ref), "\n".join(hyp))
                rouge1 += score["rouge1"].fmeasure
                rouge2 += score["rouge2"].fmeasure
                rougeLsum += score["rougeLsum"].fmeasure
                # 计算FactCC
                factcc_input_dict = factcc_tok(source_article, _hypothesis, max_length=512, padding='max_length',
                                               truncation='only_first', return_tensors='pt')
                factcc_logits = torch.softmax(factcc_model(**(factcc_input_dict.to(device))).logits, dim=1)
                factcc += factcc_logits[0][0].item()
                # 计算qags
                qags += factsumm.extract_qas(source_article, _hypothesis, f'cuda:0')

            rouge1 = rouge1 / total_num
            rouge2 = rouge2 / total_num
            rougeLsum = rougeLsum / total_num
            factcc = factcc / total_num
            qags = qags / total_num
            print("evaluating rouge1: %.4f, rouge2: %.4f, rougeL: %.4f, factCC: %.4f, QAGS: %.4f" % (
                rouge1 * 100, rouge2 * 100, rougeLsum * 100, factcc * 100, qags * 100))


def totest(gen_dataloader, model, args, tok, gpuid, do_sample=False):
    """构造用于FactCC指标计算的模型和分词器"""
    factcc_model = BertForSequenceClassification.from_pretrained("huggingface/factCC2").to(f"cuda:{gpuid}")
    factcc_tok = BertTokenizer.from_pretrained("huggingface/factCC2")

    if cuda:
        device = f"cuda:{gpuid}"
    else:
        device = "cpu"
    if len(gpuids) > 1:
        _model = model.module
    else:
        _model = model
    rouge_scorer = RougeScorer(['rouge1', 'rouge2', 'rougeLsum'], use_stemmer=True)

    _model.eval()
    """对生成器生成的样本进行评估"""
    cnt = 0
    sample_rouge1, sample_rouge2, sample_rougelsum, sample_factcc = 0, 0, 0, 0
    if do_sample:
        # 生成模式
        _model.generation_mode()

        def process(sentence):
            return sent_tokenize(" ".join(word_tokenize(sentence.strip())))

        with torch.no_grad():
            for (i, batch) in tqdm(enumerate(gen_dataloader), total=len(gen_dataloader), ascii=True,
                                   desc="对模型生成的摘要评估", position=0, colour="#99FFFF"):
                if cuda:
                    to_cuda(batch, device)
                samples = batch["data"]
                slines = [" ".join(x["article_untok"]).strip() for x in samples]
                dct = tok.batch_encode_plus(slines, max_length=args.total_len, return_tensors="pt",
                                            pad_to_max_length=True, truncation=True)
                summaries = _model.model.generate(
                    input_ids=dct["input_ids"].to(device),
                    attention_mask=dct["attention_mask"].to(device),
                    max_length=args.gen_max_len + 2,
                    min_length=args.gen_min_len + 1,
                    no_repeat_ngram_size=3,
                    num_beams=args.num_beams,
                    length_penalty=args.length_penalty,
                    early_stopping=True)
                dec = [tok.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=False) for g in
                       summaries]
                _model.scoring_mode()
                for (hypothesis, x) in zip(dec, samples):
                    hypothesis = hypothesis.replace("\n", " ").strip()
                    ref = " ".join(x["abstract_untok"]).strip()
                    source = " ".join(x["article_untok"]).strip()
                    x = process(ref)
                    y = process(hypothesis)

                    # 计算Rouge
                    score = rouge_scorer.score("\n".join(x), "\n".join(y))
                    sample_rouge1 += score["rouge1"].fmeasure
                    sample_rouge2 += score["rouge2"].fmeasure
                    sample_rougelsum += score["rougeLsum"].fmeasure
                    # 计算FactCC
                    factcc_input_dict = factcc_tok(source, hypothesis, max_length=512, padding='max_length',
                                                   truncation='only_first', return_tensors='pt')
                    factcc_logits = torch.softmax(factcc_model(**(factcc_input_dict.to(device))).logits, dim=1)
                    sample_factcc += factcc_logits[0][0].item()
                    cnt += 1
                    _model.generation_mode()  # 回到生成模式
        print(f"cnt={cnt}, batch num={len(gen_dataloader) * args.batch_size}")
        _model.scoring_mode()  # 回到scoring模式
        sample_rouge1 = sample_rouge1 / cnt
        sample_rouge2 = sample_rouge2 / cnt
        sample_rougelsum = sample_rougelsum / cnt
        sample_factcc = sample_factcc * 100 / cnt
        if len(gpuids) > 1:
            sample_rouge1 = torch.FloatTensor([sample_rouge1]).to(device)
            dist.all_reduce(sample_rouge1, op=dist.reduce_op.SUM)
            sample_rouge1 = sample_rouge1.item() / len(gpuids)
            sample_rouge2 = torch.FloatTensor([sample_rouge2]).to(device)
            dist.all_reduce(sample_rouge2, op=dist.reduce_op.SUM)
            sample_rouge2 = sample_rouge2.item() / len(gpuids)
            sample_rougelsum = torch.FloatTensor([sample_rougelsum]).to(device)
            dist.all_reduce(sample_rougelsum, op=dist.reduce_op.SUM)
            sample_rougelsum = sample_rougelsum.item() / len(gpuids)
            sample_factcc = torch.FloatTensor([sample_factcc]).to(device)
            dist.all_reduce(sample_factcc, op=dist.reduce_op.SUM)
            sample_factcc = sample_factcc.item() / len(gpuids)
        model.train()
    return {
        "sample_rouge1": sample_rouge1,
        "sample_rouge2": sample_rouge2,
        "sample_rougeLsum": sample_rougelsum,
        "sample_factcc": sample_factcc,
    }


def train(rank, args):
    if config == "cnndm":
        cnndm_setting(args)
    else:
        xsum_setting(args)
    # 任务初始化
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    gpuid = gpuids[rank]
    is_master = rank == 0  # 当前进程是否为分布式训练下的主进程
    is_mp = len(gpuids) > 1  # 是否为分布式训练
    world_size = len(gpuids)  # 训练时用到的cuda块数
    if is_master:
        the_id = len(os.listdir("cache"))
        recorder = Recorder(the_id, log)

    # 构造dataloader
    if args.is_pegasus:
        tok = PegasusTokenizer.from_pretrained(args.model_type)
    else:
        tok = BartTokenizer.from_pretrained(args.model_type)
    collate_fn = partial(collate_mp_fact_sum_gan, pad_token_id=tok.pad_token_id, is_test=False)  # 训练用collate_fn
    collate_fn_val = partial(collate_mp_fact_sum_gan, pad_token_id=tok.pad_token_id, is_test=True)  # 评估用collate_fn
    train_set = FactSumGANDataset(f"examples/processed_data/{args.dataset}/train",
                                  args.model_type, max_len=args.max_len, max_num=args.max_num,
                                  total_len=args.total_len, is_pegasus=args.is_pegasus,
                                  ranking_metric=args.ranking_metric)
    test_set = FactSumGANDataset(f"examples/processed_data/{args.dataset}/test",
                                 args.model_type, is_test=True, max_len=512, is_sorted=False,
                                 max_num=args.max_num, total_len=args.total_len, is_pegasus=args.is_pegasus,
                                 ranking_metric=args.ranking_metric)
    if is_mp:
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            train_set, num_replicas=world_size, rank=rank, shuffle=True)
        dataloader = DataLoader(train_set, batch_size=args.batch_size, shuffle=False, num_workers=4,
                                collate_fn=collate_fn, sampler=train_sampler)
        test_sampler = torch.utils.data.distributed.DistributedSampler(test_set, num_replicas=world_size, rank=rank)
        test_gen_dataloader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=4,
                                         collate_fn=collate_fn_val,
                                         sampler=test_sampler)
    else:
        dataloader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=4,
                                collate_fn=collate_fn)
        test_gen_dataloader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=4,
                                         collate_fn=collate_fn_val)

    # 构造模型
    model_path = args.pretrained if args.pretrained is not None else args.model_type
    model = Generator(model_path, tok.pad_token_id, is_pegasus=args.is_pegasus)
    if len(model_pt) > 0:
        model.load_state_dict(torch.load(os.path.join("cache/gen", model_pt), map_location=f'cuda:{gpuid}'))

    if cuda:
        if is_mp:
            # 进行分布式训练
            dist.init_process_group("nccl", rank=rank, world_size=world_size)
            model = nn.parallel.DistributedDataParallel(model.to(gpuid), [gpuid], find_unused_parameters=False)

        else:
            model = model.to(f"cuda:{gpuid}")
    model.train()

    # 将模型置为scoring模式，表示要计算对摘要赋予的概率
    if is_mp:
        model.module.scoring_mode()
    else:
        model.scoring_mode()
    if is_master:
        recorder.write_config(args, [model], __file__)

    minimum_mle_loss = 1e5  # 记录最小mle损失
    minimum_factcc_loss = 100  # 记录最小factcc事实一致性损失

    if is_mp:
        if is_master:
            the_id = torch.FloatTensor([the_id]).to(f'cuda:{gpuid}')
        else:
            the_id = torch.zeros(1).to(f'cuda:{gpuid}')
        dist.all_reduce(the_id, op=dist.reduce_op.SUM)
        the_id = int(the_id.item())

    # 根据不同的数据集调整Rouge损失计算
    if args.dataset.find("xsum") != -1:
        def eval_fn(rouge1, rouge2, rougeLsum):
            return 1 - 2 * rouge1 * rouge2 / (rouge1 + rouge2)
    else:
        def eval_fn(rouge1, rouge2, rougeLsum):
            return 1 - (rouge1 * rouge2 + rougeLsum) / 3

    # 定义优化器
    gen_optimizer = optim.Adam(model.parameters(), lr=args.max_lr)
    gen_all_step_cnt = 0  # 训练步骤数累积到args.accumulate_step的次数，即更新参数的总次数
    batch_num = args.report_freq * args.accumulate_step
    # 开始训练
    for epoch in range(args.epoch):
        gen_optimizer.zero_grad()
        pg_score_sum = 0
        ranking_loss_sum = 0
        score_sum = 0
        rougel_sum = 0
        prob_sum = torch.zeros(args.max_num)
        gold_prob_sum = 0
        gen_step_cnt = 0
        gen_epoch_step = 0  # 一个epoch内生成器更新参数的次数

        for (i, batch) in tqdm(enumerate(dataloader), total=len(dataloader), ascii=True, desc="训练"):
            if cuda:
                to_cuda(batch, f'cuda:{gpuid}')
            gen_step_cnt += 1
            # 前置step1：生成摘要
            model.generation_mode()
            with torch.no_grad():
                summaries = model.model.generate(
                    input_ids=batch["src_pad_max_input_ids"],
                    attention_mask=batch["src_pad_max_attmasks"],
                    max_length=args.gen_max_len + 2,
                    # +2 from original because we start at step=1 and stop before max_length
                    min_length=args.gen_min_len + 1,  # +1 from original because we start at step=1
                    no_repeat_ngram_size=3,
                    num_beams=args.num_beams,
                    length_penalty=args.length_penalty,
                    early_stopping=True
                )
            model.scoring_mode()
            dec = [tok.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=False) for g in summaries]

            """-------------------------计算损失和奖励-------------------------"""
            #  计算rouge奖励
            rouge_scores = []
            rouge_scorer = RougeScorer(['rouge1', 'rouge2', 'rougeLsum'], use_stemmer=True)
            for ref, hyp in zip(batch["gold"], dec):
                rouge_score = rouge_scorer.score("\n".join(sent_tokenize(" ".join(word_tokenize(ref.strip())))),
                                                 "\n".join(sent_tokenize(
                                                     " ".join(word_tokenize(hyp.replace('\n', " ").strip())))))
                rouge_scores.append(rouge_score["rougeLsum"].fmeasure)
            rouge_scores = torch.tensor(rouge_scores).data
            rougel_sum += torch.sum(rouge_scores).item()
            # rouge_scores = torch.clamp(rouge_scores - 0.29, min=0)  # 带base的奖励

            # 计算对比排序损失
            output = model(batch["src_input_ids"], batch["candidate_ids"], args.normalize, args.score_mode,
                           args.length_penalty, adding=args.adding)
            similarity, gold_similarity = output['score'], output['summary_score']
            similarity = similarity * args.scale
            gold_similarity = gold_similarity * args.scale
            ranking_loss = RankingLoss(similarity, gold_similarity, args.margin, args.gold_margin, args.gold_weight)
            ranking_loss_sum += ranking_loss.item()
            similarity = torch.exp(similarity)
            gold_similarity = torch.exp(gold_similarity)
            gold_prob_sum += torch.sum(gold_similarity.data).item()
            prob_sum += torch.sum(similarity.data, dim=0).cpu()

            # 计算策略梯度
            gen_input_ids = tok.batch_encode_plus(dec, max_length=args.total_len, return_tensors="pt",
                                                  padding='max_length', truncation=True)
            gen_pg = model.batchPG(batch["src_input_ids"], gen_input_ids["input_ids"].to(f"cuda:{gpuid}"),
                                   rouge_scores)
            pg_score_sum += gen_pg.item()

            # 计算综合得分并更新梯度
            rank_weight = 1 if gen_all_step_cnt > 1250 else args.rank_weight
            train_target = -ranking_loss * rank_weight + gen_pg * args.pg_weight  # 以梯度上升的方式极大化
            score_sum += train_target.item()
            train_target = train_target * (-1)
            train_target.backward()

            """当训练达到某种程度时进行相应处理"""
            if gen_step_cnt == args.accumulate_step:
                # 更新
                if args.grad_norm > 0:
                    nn.utils.clip_grad_norm_(model.parameters(), args.grad_norm)
                gen_step_cnt = 0
                gen_epoch_step += 1
                gen_all_step_cnt += 1

                # 调整学习率
                lr = args.max_lr * min(gen_all_step_cnt ** (-0.5), gen_all_step_cnt * (args.warmup_steps ** (-1.5)))
                for param_group in gen_optimizer.param_groups:
                    param_group['lr'] = lr
                gen_optimizer.step()
                gen_optimizer.zero_grad()
            if gen_epoch_step % args.report_freq == 0 and gen_step_cnt == 0 and is_master:
                # 汇报生成器当前状态
                print("id: %d" % the_id)
                print(
                    f"avg rouge-L: {(rougel_sum / (batch_num * args.batch_size)):.4f} ,avg probs: {prob_sum / (batch_num * args.batch_size)}")
                if not args.no_gold:
                    print(f"avg gold prob: {(gold_prob_sum / (batch_num * args.batch_size)):.4f}")
                recorder.print(
                    "traning: epoch: %d, batch: %d, avg pg score: %.6f, avg ranking loss: %.6f, avg score: %.6f" % (
                        epoch + 1, gen_epoch_step, pg_score_sum / (args.report_freq * args.accumulate_step),
                        ranking_loss_sum / (args.report_freq * args.accumulate_step),
                        score_sum / (args.report_freq * args.accumulate_step)))
                recorder.print(f"learning rate: {lr:.6f}")
                recorder.plot("score sum",
                              {"comprehensive score": score_sum / (args.report_freq * args.accumulate_step)},
                              gen_all_step_cnt)
                recorder.plot("ranking_loss",
                              {"ranking_loss": ranking_loss_sum / (args.report_freq * args.accumulate_step)},
                              gen_all_step_cnt)
                recorder.plot("pg", {"pg_score_sum": pg_score_sum / (args.report_freq * args.accumulate_step)},
                              gen_all_step_cnt)
                recorder.print()

                rougel_sum = 0
                gold_prob_sum = 0
                prob_sum = torch.zeros(args.max_num)
                pg_score_sum = 0
                ranking_loss_sum = 0
                score_sum = 0
            # 释放一些变量

            del gen_input_ids, gen_pg, ranking_loss, train_target, output, similarity, gold_similarity

            if gen_all_step_cnt % args.eval_interval == 0 and gen_all_step_cnt != 0 and gen_step_cnt == 0:
                # if gen_all_step_cnt > 10000:
                #     # 评估模型当前的效果
                #     result = totest(test_gen_dataloader, model, args, tok, gpuid, args.do_sample)
                #     # 作为生成器评估
                #     if args.do_sample:
                #         mle_loss = eval_fn(result["sample_rouge1"], result["sample_rouge2"], result["sample_rougeLsum"])
                #         factcc_loss = 100 - result["sample_factcc"]
                #     else:
                #         mle_loss = result["mle_loss"]
                #         factcc_loss = 100 - result["sample_factcc"]
                #     if mle_loss < minimum_mle_loss and is_master:
                #         minimum_mle_loss = mle_loss
                #         model_generation_name = f"gen_FC_{int(result['sample_factcc'] * 100)}_r1_{int(result['sample_rouge1'] * 10000)}_r2_{int(result['sample_rouge2'] * 10000)}_rl_{int(result['sample_rougeLsum'] * 10000)}.bin"
                #         if is_mp:
                #             recorder.save(model.module, model_generation_name)
                #         else:
                #             recorder.save(model, model_generation_name)
                #         recorder.print(
                #             "best generation loss - epoch: %d, batch: %d" % (epoch, i / args.accumulate_step))
                #     if factcc_loss < minimum_factcc_loss and is_master:
                #         minimum_factcc_loss = factcc_loss
                #         model_generation_name = f"fact_FC_{int(result['sample_factcc'] * 100)}_r1_{int(result['sample_rouge1'] * 10000)}_r2_{int(result['sample_rouge2'] * 10000)}_rl_{int(result['sample_rougeLsum'] * 10000)}.bin"
                #         if is_mp:
                #             recorder.save(model.module, model_generation_name)
                #         else:
                #             recorder.save(model, model_generation_name)
                #         recorder.print(
                #             "best factCC loss - epoch: %d, batch: %d" % (epoch, i / args.accumulate_step))
                #     if is_master:
                #         recorder.print("val generation loss: %.6f" % mle_loss)
                #         if args.do_sample:
                #             recorder.print("val generation r1: %.6f, r2: %.6f, rL: %.6f, factCC：%.6f"
                #                            % (result["sample_rouge1"] * 100, result["sample_rouge2"] * 100,
                #                               result["sample_rougeLsum"] * 100, result["sample_factcc"]))

                # 保存当前模型
                if is_master:
                    # model_temp_name = f"temp_FC_{int(result['sample_factcc'] * 100)}_r1_{int(result['sample_rouge1'] * 10000)}_r2_{int(result['sample_rouge2'] * 10000)}_rl_{int(result['sample_rougeLsum'] * 10000)}.bin"
                    model_temp_name = f"sample_num-{gen_all_step_cnt * args.accumulate_step}.bin"
                    if is_mp:
                        recorder.save(model.module, model_temp_name)
                    else:
                        recorder.save(model, model_temp_name)
                    recorder.save(gen_optimizer, "optimizer.bin")


def main(args):
    # set env
    if len(gpuids) > 1:
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = f'{port}'
        mp.spawn(train, args=(args,), nprocs=len(gpuids), join=True)
    else:
        train(0, args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Parameters')
    # parser.add_argument("--cuda", action="store_true", help="use cuda")
    # parser.add_argument("--gpuid", nargs='+', type=int, default=0, help="gpu ids")
    # parser.add_argument("-e", "--evaluate", action="store_true", help="evaluate model")
    # parser.add_argument("-r", "--do_reranking", action="store_true", help="do reranking evaluation")
    # parser.add_argument("-g", "--do_generation", action="store_true", help="do generation evaluation")
    # parser.add_argument("-l", "--log", action="store_true", help="logging")
    # parser.add_argument("-p", "--port", type=int, default=12355, help="port")
    # parser.add_argument("--model_pt", default="", type=str, help="model path")
    # parser.add_argument("--discriminator_pt", default="", type=str, help="discriminator path")
    # parser.add_argument("--ranked_metric", default="factcc", type=str, help="ranked_metric used in CL")
    # parser.add_argument("--config", default="", type=str, help="config path")  # config配置是cnn_dm的还是xsum的
    args = parser.parse_args()
    if cuda is False:
        if evaluate:
            to_evaluation(args)
        else:
            main(args)
    else:
        if evaluate:
            with torch.cuda.device(gpuids[0]):
                to_evaluation(args)
        elif len(gpuids) == 1:
            with torch.cuda.device(gpuids[0]):
                main(args)
        else:
            main(args)
