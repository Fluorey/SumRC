"""把raw__split中的数据中的原文和参考摘要抽离出来，形成.source和.target文件"""
import os
from tqdm import tqdm


def get_raw_examples(cnn_dm=True):
    if cnn_dm:
        folder_paths = {'train': "raw_split/cnn_dm/train",
                        'test': "raw_split/cnn_dm/test",
                        'val': "raw_split/cnn_dm/val"}
        targer_paths = {'train': "examples/raw_data/cnn_dm/train/",
                        'test': "examples/raw_data/cnn_dm/test/",
                        'val': "examples/raw_data/cnn_dm/val/"}
    else:
        folder_paths = {'train': "raw_split/xsum/train",
                        'test': "raw_split/xsum/test",
                        'val': "raw_split/xsum/val"}
        targer_paths = {'train': "examples/raw_data/xsum/train/",
                        'test': "examples/raw_data/xsum/test/",
                        'val': "examples/raw_data/xsum/val/"}
    for split, path in folder_paths.items():
        targer_path = targer_paths[split]

        targer_path_target = os.path.join(targer_path, split + ".target")
        targer_path_source = os.path.join(targer_path, split + ".source")
        f_target = open(targer_path_target, 'w', encoding='utf-8')
        f_source = open(targer_path_source, 'w', encoding='utf-8')
        for file_name in tqdm(os.listdir(path), desc=split + "子集整理", ascii=True):
            file_path = os.path.join(path, file_name)
            flag_target = False
            flag_source = False
            flag_source_cnn_dm = True
            target = ''
            source = ''
            for line in open(file_path, encoding='utf-8', errors='ignore').readlines():
                if not cnn_dm:
                    if flag_target:
                        line = line.replace("\n", '')
                        target += line
                        flag_target = False
                    if flag_source:
                        line = line.replace("\n", '')
                        source += line + ' '
                    if line.strip() == "[SN]FIRST-SENTENCE[SN]":
                        flag_target = True
                    if line.strip() == "[SN]RESTBODY[SN]":
                        flag_source = True
                else:
                    if line.strip() == "@highlight":
                        flag_source_cnn_dm = False
                        flag_target = True
                    elif line.strip() != "":
                        result_string = line.strip()
                        cnn_start_index = result_string.find("(CNN)")
                        if cnn_start_index != -1:
                            # 截取"(CNN)"之后的部分
                            result_string = result_string[cnn_start_index + len("(CNN)") + 4:]
                        if flag_source_cnn_dm:
                            result_string = result_string.replace("\n", '')
                            source += result_string + ' '
                        if flag_target:
                            result_string = result_string.replace("\n", '')
                            target += result_string + '. '
            if source != '' and target != '':
                f_target.write(target + '\n')
                f_target.flush()
                f_source.write(source + '\n')
                f_source.flush()
        f_target.close()
        f_source.close()
