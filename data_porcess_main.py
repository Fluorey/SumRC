import os
from data_process.raw_data_split import raw_data_split
from data_process.get_raw_examles import get_raw_examples
from generate_candidates import generate_summaries_xsum, generate_summaries_cnndm
from transformers import BartForConditionalGeneration, BartTokenizer, PegasusTokenizer, PegasusForConditionalGeneration
from tqdm import tqdm

from stanfordcorenlp import StanfordCoreNLP

nlp = StanfordCoreNLP(r'nlpcore/stanford-corenlp-4.5.5/', lang='en')

if __name__ == '__main__':
    """数据集分割"""
    # raw_data_split(data_folder="dailymail/stories")

    """将每个子集融合在一起，形成.source和.target文件"""
    # get_raw_examples(False)

    """数据检查"""
    indexx = 1279890-1
    # part = "has spent the past five weeks by the"
    # with open("examples/raw_data/cnn_dm/test/test.source", 'r', encoding='gb18030', errors='ignore') as file:
    #     line_count1 = sum(1 for line in file)
    # with open("examples/raw_data/cnn_dm/test/test.source", 'r', encoding='gb18030', errors='ignore') as file, open(
    #         "examples/raw_data/cnn_dm/test/test.source.tokenized", 'w') as fout:
    #     # for index, line in enumerate(file):
    #     #     if index == indexx:
    #     #         print("原未分词：", line)
    #     # for index, line in enumerate(file):
    #     #     if part in line:
    #     #         print(index, line)
    #     for line in tqdm(file, total=line_count1, desc='分词处理', ascii=True):
    #         fout.write(' '.join(nlp.word_tokenize(line.strip())) + '\n')
    #         fout.flush()
    #     # for index, line in tqdm(enumerate(file), total=line_count1, desc='分词处理', ascii=True):
    #     #     if index == indexx or index == indexx + 1 or index == indexx + 2 or index == indexx + 3 or index == indexx + 4 or index == indexx + 5:
    #     #         continue
    #     #     fout.write(line.strip() + '\n')
    #     #     fout.flush()
    # fout.close()

    with open("examples/raw_data/xsum/train/train.out", 'r', encoding='utf-8', errors='ignore') as file:
        for index, line in enumerate(file):
            if index == indexx:
                print(line)
    with open("examples/raw_data/xsum/train/train.out.tokenized", 'r', encoding='utf-8', errors='ignore') as file:
        for index, line in enumerate(file):
            if index == indexx:
                print(line)

    # with open("examples/raw_data/xsum/train/train.source", 'r', encoding='utf-8', errors='ignore') as file:
    #     line_count1 = sum(1 for line in file)
    # with open("examples/raw_data/xsum/train/train.target", 'r', encoding='utf-8', errors='ignore') as file:
    #     line_count2 = sum(1 for line in file)
    # with open("examples/raw_data/xsum/train/train.out", 'r', encoding='utf-8', errors='ignore') as file:
    #     line_count3 = sum(1 for line in file)
    # with open("examples/raw_data/xsum/train/train.source.tokenized", 'r') as file:
    #     line_count4 = sum(1 for line in file)
    # with open("examples/raw_data/xsum/train/train.target.tokenized", 'r', encoding='utf-8', errors='ignore') as file:
    #     line_count5 = sum(1 for line in file)
    # with open("examples/raw_data/xsum/train/train.out.tokenized", 'r', encoding='utf-8', errors='ignore') as file:
    #     line_count6 = sum(1 for line in file)
    # print(line_count1, line_count2, line_count3, '\n', line_count4, line_count5, line_count3)

    """生成候选摘要"""
    # is_cnndm = False
    # gpuid = [0, 1, 2, 3]
    # # if is_cnndm:
    # #     scr_dirs = {'train': "examples/raw_data/cnn_dm/train/train.source",
    # #                 'test': "examples/raw_data/cnn_dm/test/test.source",
    # #                 'val': "examples/raw_data/cnn_dm/val/val.source"}
    # #     tgr_dirs = {'train': "examples/raw_data/cnn_dm/train/train.out",
    # #                 'test': "examples/raw_data/cnn_dm/test/test.out",
    # #                 'val': "examples/raw_data/cnn_dm/val/val.out"}
    # # else:
    # #     scr_dirs = {'train': "examples/raw_data/xsum/train/train.source",
    # #                 'test': "examples/raw_data/xsum/test/test.source",
    # #                 'val': "examples/raw_data/xsum/val/val.source"}
    # #     tgr_dirs = {'train': "examples/raw_data/xsum/train/train.out",
    # #                 'test': "examples/raw_data/xsum/test/test.out",
    # #                 'val': "examples/raw_data/xsum/val/val.out"}
    # if is_cnndm:
    #     scr_dirs = {'train': "examples/raw_data/cnn_dm/train/train.source",
    #                 }
    #     tgr_dirs = {'train': "examples/raw_data/cnn_dm/train/train.out",
    #                 }
    # else:
    #     scr_dirs = {'train': "examples/raw_data/xsum/train/train.source",
    #                 }
    #     tgr_dirs = {'train': "examples/raw_data/xsum/train/train.out",
    #                 }
    # for split, path in scr_dirs.items():
    #     if is_cnndm:
    #         generate_summaries_cnndm(path, tgr_dirs[split], gpuid=gpuid)
    #     else:
    #         generate_summaries_xsum(path, tgr_dirs[split], gpuid=gpuid)
