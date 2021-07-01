# !/user/bin/python
# -*- coding:utf-8 -*-
"""
date：          2021-06-30
Description :
auther : wcy
"""
# import modules
import os

__all__ = ["Config"]


# config
class Config:
    k = 50
    conf_th = 0.7

    NUM_CLASSES = 11014
    NUM_WORKERS = 2
    SEED = 0

    test_size = 384

    # image model backbone
    image_backbone_model_1 = 'vit_deit_base_distilled_patch16_384'
    image_feature_model1_fc_dim = 768
    image_feature_model1_p_eval = 6.0

    image_backbone_model_2 = 'dm_nfnet_f0'
    image_feature_model2_fc_dim = 256
    image_feature_model2_p_eval = 6.0

    # multi modal
    multi_modal_image_backbone_model = 'dm_nfnet_f0'
    multi_modal_text_max_len = 64
    multi_modal_image_fc_dim = 1024
    multi_modal_s = 50
    multi_modal_margin = 0.3
    multi_modal_loss = 'CurricularFace'
    multi_modal_p_eval = 6.0

    # bert
    bert_vocab_file = '/2021-top-data-competition/1-shopee同款商品选择最低价/_2nd_place/pretrained_model/author_trained/bert-indo/vocab.txt'
    assert os.path.exists(bert_vocab_file)
    bert_config_file = '/2021-top-data-competition/1-shopee同款商品选择最低价/_2nd_place/pretrained_model/author_trained/bert-indo/config.json'
    assert os.path.exists(bert_config_file)

    bert_batch_size = 128
    bert_max_len = 64
    bert_fc_dim = 512
    bert_s = 50
    bert_margin = 0.3
    bert_loss = 'CurricularFace'

    # bert 2
    bert2_pretrained_path = '/2021-top-data-competition/1-shopee同款商品选择最低价/_2nd_place/pretrained_model/author_trained/bert-multilingual'
    assert os.path.exists(bert2_pretrained_path)

    bert2_model_name = 'bert-base-multilingual-uncased'
    bert2_max_len = 64
    bert2_fc_dim = 512
    bert2_s = 50
    bert2_margin = 0.3
    bert2_loss = 'CurricularFace'

    # bert 3
    bert3_model_ckpt = '/2021-top-data-competition/1-shopee同款商品选择最低价/_2nd_place/pretrained_model/author_trained/bert-xlm'
    assert os.path.exists(bert3_model_ckpt)

    bert3_pretrained_path = 'sentence-transformers/paraphrase-xlm-r-multilingual-v1'
    bert3_max_len = 64
    bert3_fc_dim = 256
    bert3_s = 30
    bert3_margin = 0.5
    bert3_loss = 'CurricularFace'

    def __init__(self, data_path, save_dir):
        self.data_path = data_path
        self.save_dir = save_dir
        assert os.path.exists(save_dir)
        assert len(os.listdir(save_dir)) == 0

