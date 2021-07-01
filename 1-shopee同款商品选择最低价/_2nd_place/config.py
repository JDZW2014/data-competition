# !/user/bin/python
# -*- coding:utf-8 -*-
"""
date：          2021-06-30
Description :
auther : wcy
"""
# import modules
import pandas as pd


__all__ = ["Config"]


# config
class Config:

    k = 50
    conf_th = 0.7
    DEBUG = len(pd.read_csv('../input/shopee-product-matching/test.csv')) == 3

    NUM_CLASSES = 11014
    NUM_WORKERS = 2
    SEED = 0

    image_size = ''
    # image model backbone
    image_backbone_model_1 = ""
    image_feature_model1_fc_dim = ''
    image_feature_model1_p_eval = ''

    image_backbone_model_2 = ""
    image_feature_model2_fc_dim = ''
    image_feature_model2_p_eval = ''

    # multi modal
    multi_modal_image_backbone_model = ''
    multi_modal_image_fc_dim = ''
    multi_modal_image_p_eval = ''

    # bret
    bert_vocab_file = ''
    bert_config_file = ''
    text_max_len = ''
    bert_fc_dim = ''
    bert_p_eval = ''
    bert_batch_size = ''
    bert_model_ckpt = ''
