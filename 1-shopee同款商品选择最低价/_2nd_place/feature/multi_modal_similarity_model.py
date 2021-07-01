# !/user/bin/python
# -*- coding:utf-8 -*-
"""
date：          2021-06-30
Description :
auther : wcy
"""
# import modules
import torch
from torch import nn
import timm
from transformers import BertConfig, BertModel, BertTokenizerFast
from _2nd_place.utils import gem

__all__ = ["MultiModalNet", "create_model"]


# define class
class MultiModalNet(nn.Module):

    def __init__(self, backbone, bert_model, tokenizer, max_len=32, fc_dim=512, p=3):
        super().__init__()

        self.backbone = backbone
        self.backbone.reset_classifier(num_classes=0)  # remove classifier

        self.bert_model = bert_model
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.fc = nn.Linear(self.bert_model.config.hidden_size + self.backbone.num_features, fc_dim)
        self.bn = nn.BatchNorm1d(fc_dim)
        self._init_params()
        self.p = p

    def _init_params(self):
        nn.init.xavier_normal_(self.fc.weight)
        nn.init.constant_(self.fc.bias, 0)
        nn.init.constant_(self.bn.weight, 1)
        nn.init.constant_(self.bn.bias, 0)

    def extract_feat(self, img, title):
        batch_size = img.shape[0]
        img = self.backbone.forward_features(img)
        img = gem(img, p=self.p).view(batch_size, -1)

        tokenizer_output = self.tokenizer(title, truncation=True, padding=True, max_length=self.max_len)
        input_ids = torch.LongTensor(tokenizer_output['input_ids']).to('cuda')
        token_type_ids = torch.LongTensor(tokenizer_output['token_type_ids']).to('cuda')
        attention_mask = torch.LongTensor(tokenizer_output['attention_mask']).to('cuda')
        title = self.bert_model(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
        # x = x.last_hidden_state.sum(dim=1) / attention_mask.sum(dim=1, keepdims=True)
        title = title.last_hidden_state.mean(dim=1)

        x = torch.cat([img, title], dim=1)
        x = self.fc(x)
        x = self.bn(x)
        return x


# define function
def create_model(model_name, bert_vocab_file, bert_config_file, max_len, fc_dim, p,
                 to_cuda=False, model_ckpt=None, if_train=False):

    backbone = timm.create_model(model_name=model_name, pretrained=False)
    # tokenizer = BertTokenizerFast(vocab_file='../input/bert-indo/vocab.txt')
    tokenizer = BertTokenizerFast(vocab_file=bert_vocab_file)
    # bert_config = BertConfig.from_json_file('../input/bert-indo/config.json')
    bert_config = BertConfig.from_json_file(bert_config_file)
    bert_model = BertModel(bert_config)
    model = MultiModalNet(backbone, bert_model, tokenizer=tokenizer, max_len=max_len, fc_dim=fc_dim)
    model.p = p

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
