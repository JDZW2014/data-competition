# !/user/bin/python
# -*- coding:utf-8 -*-
"""
date：          2021-06-30
Description :
auther : wcy
"""
# import modules
import torch
import torch.nn as nn
from transformers import BertConfig, BertModel, BertTokenizerFast
from transformers import AutoTokenizer, AutoModel, AutoConfig

__all__ = ["BertNet", "create_model_1", "create_model_2", "train_model"]


# define class
class BertNet(nn.Module):

    def __init__(self, bert_model, tokenizer, max_len=32, fc_dim=512, simple_mean=True, p=3):
        super().__init__()

        self.bert_model = bert_model
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.fc = nn.Linear(self.bert_model.config.hidden_size, fc_dim)
        self.bn = nn.BatchNorm1d(fc_dim)
        self._init_params()
        self.p = p
        self.simple_mean = simple_mean

    def _init_params(self):
        nn.init.xavier_normal_(self.fc.weight)
        nn.init.constant_(self.fc.bias, 0)
        nn.init.constant_(self.bn.weight, 1)
        nn.init.constant_(self.bn.bias, 0)

    def extract_feat(self, x, to_cuda=False):
        tokenizer_output = self.tokenizer(x, truncation=True, padding=True, max_length=self.max_len)
        if 'token_type_ids' in tokenizer_output:
            input_ids = torch.LongTensor(tokenizer_output['input_ids'])
            token_type_ids = torch.LongTensor(tokenizer_output['token_type_ids'])
            attention_mask = torch.LongTensor(tokenizer_output['attention_mask'])
            if to_cuda:
                input_ids = input_ids.to('cuda')
                token_type_ids = token_type_ids.to('cuda')
                attention_mask = attention_mask.to('cuda')

            x = self.bert_model(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
        else:
            input_ids = torch.LongTensor(tokenizer_output['input_ids'])
            attention_mask = torch.LongTensor(tokenizer_output['attention_mask'])
            if to_cuda:
                input_ids = input_ids.to('cuda')
                attention_mask = attention_mask.to('cuda')
            x = self.bert_model(input_ids=input_ids, attention_mask=attention_mask)
        if self.simple_mean:
            x = x.last_hidden_state.mean(dim=1)
        else:
            x = torch.sum(x.last_hidden_state * attention_mask.unsqueeze(-1), dim=1) / attention_mask.sum(dim=1, keepdims=True)
        x = self.fc(x)
        x = self.bn(x)
        return x


# define function
def create_model_1(vocab_file_path, bert_config_file, max_len, fc_dim,
                   to_cuda=False, model_ckpt=None, if_train=False, simple_mean=True):
    tokenizer = BertTokenizerFast(vocab_file=vocab_file_path)
    bert_config = BertConfig.from_json_file(bert_config_file)
    bert_model = BertModel(bert_config)
    model = BertNet(bert_model, tokenizer=tokenizer, max_len=max_len, simple_mean=simple_mean, fc_dim=fc_dim)

    if to_cuda:
        model = model.to('cuda')
    if model_ckpt is not None:
        model.load_state_dict(model_ckpt, strict=False)
    if if_train is False:
        model.train(False)
    return model


def create_model_2(pretrained_path, max_len, fc_dim,
                   to_cuda=False, model_ckpt=None, if_train=False, simple_mean=False):
    print("--", pretrained_path, "--")
    tokenizer = AutoTokenizer.from_pretrained(pretrained_path)
    bert_config = AutoConfig.from_pretrained(pretrained_path)
    bert_model = AutoModel.from_config(bert_config)
    model = BertNet(bert_model, tokenizer=tokenizer, max_len=max_len, simple_mean=simple_mean, fc_dim=fc_dim)

    if to_cuda:
        model = model.to('cuda')
    if model_ckpt is not None:
        model.load_state_dict(model_ckpt, strict=False)
    if if_train is False:
        model.train(False)
    return model


def train_model():
    pass


# main
if __name__ == '__main__':
    pass
