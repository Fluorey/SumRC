import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel


class Discriminator(nn.Module):

    def __init__(self, dropout=0.5, bert_model_name='huggingface/bert-large-uncased'):
        super(Discriminator, self).__init__()
        self.bert = BertModel.from_pretrained(bert_model_name)
        self.threshold = 0.5
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(1024, 2)

    def forward(self, real_text_ids, gen_text_ids, real_text_attmasks, gen_text_attmasks):
        """
        real_text_ids: [bz,seq_len]
        gen_text_ids: [bz,seq_len]
        """
        """获取outputs"""
        real_outputs = self.bert(input_ids=real_text_ids, attention_mask=real_text_attmasks)
        gen_outputs = self.bert(input_ids=gen_text_ids, attention_mask=gen_text_attmasks)
        """转换为概率"""
        real_final_layer = F.relu(self.linear(self.dropout(real_outputs.pooler_output)))  # 获取正类在最后线性层上的结果
        gen_final_layer = F.relu(self.linear(self.dropout(gen_outputs.pooler_output)))  # 获取负类在最后线性层上的结果
        real_probs = torch.softmax(real_final_layer, dim=1)[:, 1]  # 获取正类样本被判别器归为正类的概率:[bz]
        gen_probs = torch.softmax(gen_final_layer, dim=1)[:, 1]  # 获取负类样本被判别器归为正类的概率:[bz]
        """获取预测标签"""
        real_pre_labels = (real_probs >= self.threshold).long()  # 获取正类的预测标签
        gen_pre_labels = (gen_probs >= self.threshold).long()  # 获取负类的预测标签

        return {"real_probs": real_probs, "gen_probs": gen_probs, "real_pre_labels": real_pre_labels,
                "gen_pre_labels": gen_pre_labels}

    def batchNLLLoss(self, real_probs, gen_probs):
        """
        计算判别器训练时的二分类交叉熵损失
        """
        real_targets = torch.ones_like(real_probs)
        gen_targets = torch.zeros_like(gen_probs)
        loss_fn = nn.BCELoss()

        return (loss_fn(real_probs, real_targets) + loss_fn(gen_probs, gen_targets)) / 2
