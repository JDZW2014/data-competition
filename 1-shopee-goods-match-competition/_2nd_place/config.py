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
    image_batch_size = 2

    test_size = 384

    # image model backbone
    image1_trained_moel_path = "/2021-top-data-competition/1-shopee-goods-match-competition/_2nd_place/pretrained_model/author_trained/v45.pth"
    assert os.path.exists(image1_trained_moel_path)

    image_backbone_model_1 = 'vit_deit_base_distilled_patch16_384'
    image_feature_model1_fc_dim = 768
    image_feature_model1_p_eval = 6.0
    image_1_feat_save_name = "img_feats1"

    image2_trained_model_path = "/2021-top-data-competition/1-shopee-goods-match-competition/_2nd_place/pretrained_model/author_trained/v34.pth"
    assert os.path.exists(image2_trained_model_path)

    image_backbone_model_2 = 'dm_nfnet_f0'
    image_feature_model2_fc_dim = 256
    image_feature_model2_p_eval = 6.0
    image_2_feat_save_name = 'img_feats2'

    image_1_and_image_2_concat_feat_save_name = "img_feats"
    lyk_img_meta_data_save_name = "lyk_img_meta_data"

    # multi modal
    multi_modal_trained_model_path = "/2021-top-data-competition/1-shopee-goods-match-competition/_2nd_place/pretrained_model/author_trained/v79.pth"
    assert os.path.exists(multi_modal_trained_model_path)

    multi_modal_image_backbone_model = 'dm_nfnet_f0'
    multi_modal_text_max_len = 64
    multi_modal_image_fc_dim = 1024
    multi_modal_s = 50
    multi_modal_margin = 0.3
    multi_modal_loss = 'CurricularFace'
    multi_modal_p_eval = 6.0
    multi_modal_feat_save_name = "mm_feats"

    # bert
    bert_vocab_file = '/2021-top-data-competition/1-shopee-goods-match-competition/_2nd_place/pretrained_model/author_trained/bert-indo/vocab.txt'
    assert os.path.exists(bert_vocab_file)
    bert_config_file = '/2021-top-data-competition/1-shopee-goods-match-competition/_2nd_place/pretrained_model/author_trained/bert-indo/config.json'
    assert os.path.exists(bert_config_file)
    bert_trained_model_path = '/2021-top-data-competition/1-shopee-goods-match-competition/_2nd_place/pretrained_model/author_trained/v75.pth'

    bert_batch_size = 2
    bert_max_len = 64
    bert_fc_dim = 512
    bert_s = 50
    bert_margin = 0.3
    bert_loss = 'CurricularFace'
    bert_feature_save_name = "bert_feats1"

    # bert 2
    bert2_pretrained_path = '/2021-top-data-competition/1-shopee-goods-match-competition/_2nd_place/pretrained_model/author_trained/bert-multilingual'
    assert os.path.exists(bert2_pretrained_path)
    bert2_trained_model_path = '/2021-top-data-competition/1-shopee-goods-match-competition/_2nd_place/pretrained_model/author_trained/v102.pth'
    assert os.path.exists(bert2_trained_model_path)

    bert2_model_name = 'bert-base-multilingual-uncased'
    bert2_max_len = 64
    bert2_fc_dim = 512
    bert2_s = 50
    bert2_margin = 0.3
    bert2_loss = 'CurricularFace'
    bert2_feature_save_name = "bert_feats2"

    # bert 3
    bert3_pretrained_path = '/2021-top-data-competition/1-shopee-goods-match-competition/_2nd_place/pretrained_model/author_trained/bert-xlm'
    assert os.path.exists(bert3_pretrained_path)
    bert3_trained_model_path = '/2021-top-data-competition/1-shopee-goods-match-competition/_2nd_place/pretrained_model/author_trained/v103.pth'
    assert os.path.exists(bert3_trained_model_path)

    bert3_model_name = 'sentence-transformers/paraphrase-xlm-r-multilingual-v1'
    bert3_max_len = 64
    bert3_fc_dim = 256
    bert3_s = 30
    bert3_margin = 0.5
    bert3_loss = 'CurricularFace'
    bert3_feature_save_name = "bert_feats3"

    bert_model_feature_concat_save_name = "bert_feats"

    # knn search
    lyk_img_data = "lyk_img_data"
    lyk_mm_data = "lyk_mm_data"
    lyk_bert_data = "lyk_bert_data"

    def __init__(self, data_path, image_dir_path, save_dir):
        self.data_path = data_path
        self.image_dir_path = image_dir_path
        self.save_dir = save_dir
        assert os.path.exists(save_dir)
        # assert len(os.listdir(save_dir)) == 0

