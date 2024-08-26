from transformers import BartForConditionalGeneration, BartTokenizer, PegasusTokenizer, PegasusForConditionalGeneration
import torch
import argparse
from typing import List
from tqdm import tqdm


def generate_summaries_cnndm(src_dir, tgt_dir, gpuid):
    device = [f"cuda:{gpuid[0]}", f"cuda:{gpuid[1]}", f"cuda:{gpuid[2]}", f"cuda:{gpuid[3]}"]
    device_base = device[1]
    device_efactsum = device[1]
    bart_name = "huggingface/bart-large-cnn"
    efactsum_bart_name = "huggingface/efactsum-bart-cnndm"
    model_bart = BartForConditionalGeneration.from_pretrained(bart_name).to(device_base)
    model_efactsum_bart = BartForConditionalGeneration.from_pretrained(efactsum_bart_name).to(device_efactsum)
    model_bart.eval()
    model_efactsum_bart.eval()
    tokenizer_bart = BartTokenizer.from_pretrained(bart_name)
    tokenizer_efactsum = BartTokenizer.from_pretrained(efactsum_bart_name)

    max_length = 140
    min_length = 55
    count = 1
    bsz = 14
    with open(src_dir) as file:
        total_lines = sum(1 for line in file)
    with open(src_dir) as source, open(tgt_dir, 'w') as fout:
        sline = source.readline().strip().lower()
        slines = [sline]
        for sline in tqdm(source, total=total_lines, desc=tgt_dir, ascii=True):
            if count % 100 == 0:
                print(count, flush=True)
            if count % bsz == 0:
                with torch.no_grad():
                    dct_bart = tokenizer_bart.batch_encode_plus(slines, max_length=1024, return_tensors="pt",
                                                                padding='max_length', truncation=True)
                    dct_efactsum = tokenizer_efactsum.batch_encode_plus(slines, max_length=1024, return_tensors="pt",
                                                                        padding='max_length', truncation=True)
                    summaries_bart = model_bart.generate(
                        input_ids=dct_bart["input_ids"].to(device_base),
                        attention_mask=dct_bart["attention_mask"].to(device_base),
                        num_return_sequences=3, num_beam_groups=8, diversity_penalty=1.0, num_beams=8,
                        max_length=max_length + 2,
                        # +2 from original because we start at step=1 and stop before max_length
                        min_length=min_length + 1,  # +1 from original because we start at step=1
                        no_repeat_ngram_size=3,
                        length_penalty=2.0,
                        early_stopping=True,
                    )
                    summaries_efactsum = model_efactsum_bart.generate(
                        input_ids=dct_efactsum["input_ids"].to(device_efactsum),
                        attention_mask=dct_efactsum["attention_mask"].to(device_efactsum),
                        num_return_sequences=3, num_beam_groups=8, diversity_penalty=1.0, num_beams=8,
                        max_length=max_length + 2,
                        # +2 from original because we start at step=1 and stop before max_length
                        min_length=min_length + 1,  # +1 from original because we start at step=1
                        no_repeat_ngram_size=3,
                        length_penalty=2.0,
                        early_stopping=True,
                    )
                    dec_bart = [
                        tokenizer_bart.decode(g.to('cpu'), skip_special_tokens=True, clean_up_tokenization_spaces=False)
                        for g in summaries_bart]
                    dec_efactsum = [
                        tokenizer_efactsum.decode(g.to('cpu'), skip_special_tokens=True,
                                                  clean_up_tokenization_spaces=False)
                        for g in summaries_efactsum]
                for hypothesis_bart, hypothesis_efactsum in zip(dec_bart, dec_efactsum):
                    hypothesis_bart = hypothesis_bart.replace("\n", " ")
                    hypothesis_efactsum = hypothesis_efactsum.replace("\n", " ")
                    fout.write(hypothesis_bart + '\n')
                    fout.flush()
                    fout.write(hypothesis_efactsum + '\n')
                    fout.flush()
                slines = []
            sline = sline.strip().lower()
            if len(sline) == 0:
                sline = " "
            slines.append(sline)
            count += 1
        if slines:
            with torch.no_grad():
                dct_bart = tokenizer_bart.batch_encode_plus(slines, max_length=1024, return_tensors="pt",
                                                            pad_to_max_length=True, truncation=True)
                dct_efactsum = tokenizer_efactsum.batch_encode_plus(slines, max_length=1024, return_tensors="pt",
                                                                    pad_to_max_length=True, truncation=True)
                summaries_bart = model_bart.generate(
                    input_ids=dct_bart["input_ids"].to(device_base),
                    attention_mask=dct_bart["attention_mask"].to(device_base),
                    num_return_sequences=3, num_beam_groups=8, diversity_penalty=1.0, num_beams=8,
                    max_length=max_length + 2,
                    # +2 from original because we start at step=1 and stop before max_length
                    min_length=min_length + 1,  # +1 from original because we start at step=1
                    no_repeat_ngram_size=3,
                    length_penalty=2.0,
                    early_stopping=True,
                )
                summaries_efactsum = model_efactsum_bart.generate(
                    input_ids=dct_efactsum["input_ids"].to(device_efactsum),
                    attention_mask=dct_efactsum["attention_mask"].to(device_efactsum),
                    num_return_sequences=3, num_beam_groups=8, diversity_penalty=1.0, num_beams=8,
                    max_length=max_length + 2,
                    # +2 from original because we start at step=1 and stop before max_length
                    min_length=min_length + 1,  # +1 from original because we start at step=1
                    no_repeat_ngram_size=3,
                    length_penalty=2.0,
                    early_stopping=True,
                )
                dec_bart = [
                    tokenizer_bart.decode(g.to('cpu'), skip_special_tokens=True, clean_up_tokenization_spaces=False) for
                    g
                    in
                    summaries_bart]
                dec_efactsum = [
                    tokenizer_efactsum.decode(g.to('cpu'), skip_special_tokens=True, clean_up_tokenization_spaces=False)
                    for g in
                    summaries_efactsum]
            for hypothesis_bart, hypothesis_efactsum in zip(dec_bart, dec_efactsum):
                hypothesis_bart = hypothesis_bart.replace("\n", " ")
                hypothesis_efactsum = hypothesis_efactsum.replace("\n", " ")
                fout.write(hypothesis_bart + '\n')
                fout.flush()
                fout.write(hypothesis_efactsum + '\n')
                fout.flush()


def generate_summaries_xsum(src_dir, tgt_dir, gpuid):
    device = [f"cuda:{gpuid[0]}", f"cuda:{gpuid[1]}", f"cuda:{gpuid[2]}", f"cuda:{gpuid[3]}"]
    device_base = device[0]
    device_efactsum = device[0]
    pegasus_name = "huggingface/pegasus-xsum"
    efactsum_pegasus_name = "huggingface/efactsum-pegasus-xsum"
    model_pegasus = PegasusForConditionalGeneration.from_pretrained(pegasus_name).to(device_base)
    model_efactsum_pegasus = PegasusForConditionalGeneration.from_pretrained(efactsum_pegasus_name).to(device_efactsum)
    model_pegasus.eval()
    model_efactsum_pegasus.eval()
    tok_pegasus = PegasusTokenizer.from_pretrained(pegasus_name)
    tok_efactsum = PegasusTokenizer.from_pretrained(efactsum_pegasus_name)
    max_length = 62
    min_length = 11
    count = 1
    bsz = 8

    with open(src_dir) as file:
        total_lines = sum(1 for line in file)

    with open(src_dir) as source, open(tgt_dir, 'w') as fout:
        sline = source.readline().strip()
        slines = [sline]
        for sline in tqdm(source, total=total_lines, desc=tgt_dir, ascii=True):
            if count % 100 == 0:
                print(count, flush=True)
            if count % bsz == 0:
                with torch.no_grad():
                    dct_pegasus = tok_pegasus.batch_encode_plus(slines, max_length=512, return_tensors="pt",
                                                                padding='max_length', truncation=True)
                    dct_efactsum = tok_efactsum.batch_encode_plus(slines, max_length=512, return_tensors="pt",
                                                                  padding='max_length', truncation=True)
                    summaries_pegasus = model_pegasus.generate(
                        input_ids=dct_pegasus["input_ids"].to(device_base),
                        attention_mask=dct_pegasus["attention_mask"].to(device_base),
                        num_return_sequences=3, num_beam_groups=16, diversity_penalty=2.0, num_beams=16,
                        max_length=max_length + 2,
                        # +2 from original because we start at step=1 and stop before max_length
                        min_length=min_length + 1,  # +1 from original because we start at step=1
                        length_penalty=0.6,
                        no_repeat_ngram_size=3,
                        early_stopping=True,
                    )
                    summaries_efactsum = model_efactsum_pegasus.generate(
                        input_ids=dct_efactsum["input_ids"].to(device_base),
                        attention_mask=dct_efactsum["attention_mask"].to(device_base),
                        num_return_sequences=3, num_beam_groups=16, diversity_penalty=2.0, num_beams=16,
                        max_length=max_length + 2,
                        # +2 from original because we start at step=1 and stop before max_length
                        min_length=min_length + 1,  # +1 from original because we start at step=1
                        length_penalty=0.6,
                        no_repeat_ngram_size=3,
                        early_stopping=True,
                    )
                    dec_pegasus = [
                        tok_pegasus.decode(g.to('cpu'), skip_special_tokens=True, clean_up_tokenization_spaces=False)
                        for g in summaries_pegasus]
                    dec_efactsums = [
                        tok_efactsum.decode(g.to('cpu'), skip_special_tokens=True,
                                            clean_up_tokenization_spaces=False)
                        for g in summaries_efactsum]
                for hypothesis_pegasus, hypothesis_efactsums in zip(dec_pegasus, dec_efactsums):
                    hypothesis_pegasus = hypothesis_pegasus.replace("\n", " ")
                    hypothesis_efactsums = hypothesis_efactsums.replace("\n", " ")
                    fout.write(hypothesis_pegasus + '\n')
                    fout.flush()
                    fout.write(hypothesis_efactsums + '\n')
                    fout.flush()
                slines = []
            sline = sline.strip()
            if len(sline) == 0:
                sline = " "
            slines.append(sline)
            count += 1
        if slines:
            with torch.no_grad():
                dct_pegasus = tok_pegasus.batch_encode_plus(slines, max_length=512, return_tensors="pt",
                                                            padding='max_length', truncation=True)
                dct_efactsum = tok_efactsum.batch_encode_plus(slines, max_length=512, return_tensors="pt",
                                                              padding='max_length', truncation=True)
                summaries_pegasus = model_pegasus.generate(
                    input_ids=dct_pegasus["input_ids"].to(device_base),
                    attention_mask=dct_pegasus["attention_mask"].to(device_base),
                    num_return_sequences=3, num_beam_groups=16, diversity_penalty=2.0, num_beams=16,
                    max_length=max_length + 2,
                    # +2 from original because we start at step=1 and stop before max_length
                    min_length=min_length + 1,  # +1 from original because we start at step=1
                    length_penalty=0.6,
                    no_repeat_ngram_size=3,
                    early_stopping=True,
                )
                summaries_efactsum = model_efactsum_pegasus.generate(
                    input_ids=dct_efactsum["input_ids"].to(device_base),
                    attention_mask=dct_efactsum["attention_mask"].to(device_base),
                    num_return_sequences=3, num_beam_groups=16, diversity_penalty=2.0, num_beams=16,
                    max_length=max_length + 2,
                    # +2 from original because we start at step=1 and stop before max_length
                    min_length=min_length + 1,  # +1 from original because we start at step=1
                    length_penalty=0.6,
                    no_repeat_ngram_size=3,
                    early_stopping=True,
                )
                dec_pegasus = [
                    tok_pegasus.decode(g.to('cpu'), skip_special_tokens=True, clean_up_tokenization_spaces=False)
                    for g in summaries_pegasus]
                dec_efactsums = [
                    tok_efactsum.decode(g.to('cpu'), skip_special_tokens=True,
                                        clean_up_tokenization_spaces=False)
                    for g in summaries_efactsum]
            for hypothesis_pegasus, hypothesis_efactsums in zip(dec_pegasus, dec_efactsums):
                hypothesis_pegasus = hypothesis_pegasus.replace("\n", " ")
                hypothesis_efactsums = hypothesis_efactsums.replace("\n", " ")
                fout.write(hypothesis_pegasus + '\n')
                fout.flush()
                fout.write(hypothesis_efactsums + '\n')
                fout.flush()
