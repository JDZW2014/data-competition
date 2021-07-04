# !/user/bin/python
# -*- coding:utf-8 -*-
"""
date：          2021-06-30
Description :
auther : wcy
"""
# import modules
from _2nd_place.feature.get_feature import get_image_and_multi_modal_features, get_nlp_features, \
    image_knn_search, nlp_knn_search
from _2nd_place.config import Config
from _2nd_place.utils import load_author_pretrained_model_ckpt
import logging
import numpy as np
import os

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


__all__ = []


# define function
def get_feature_from_pretrained_model(config: Config, nrows, to_cuda):
    logging.info(" -- load image model and multi modal model to get feature -- ")
    image_model1_ckpt = load_author_pretrained_model_ckpt(config.image1_trained_moel_path)
    image_model2_ckpt = load_author_pretrained_model_ckpt(config.image2_trained_model_path)
    multi_modal_model_ckpt = load_author_pretrained_model_ckpt(config.multi_modal_trained_model_path)
    get_image_and_multi_modal_features(config=config, image_model1_ckpt=image_model1_ckpt,
                                       image_model2_ckpt=image_model2_ckpt,
                                       multi_modal_model_ckpt=multi_modal_model_ckpt, to_cuda=to_cuda, nrows=nrows)

    logging.info(" -- load bert model to get feature -- ")
    bert_model_ckpt = load_author_pretrained_model_ckpt(config.bert_trained_model_path)
    bert2_model_ckpt = load_author_pretrained_model_ckpt(config.bert2_trained_model_path)
    bert3_model_ckpt = load_author_pretrained_model_ckpt(config.bert3_trained_model_path)

    get_nlp_features(config=config, bert_model_ckpt=bert_model_ckpt, bert2_model_ckpt=bert2_model_ckpt,
                     bert3_model_ckpt=bert3_model_ckpt, nrows=nrows, to_cuda=to_cuda)


def fasis_first_recall(to_cuda, config):
    img_feats = np.load(os.path.join(config.save_dir, config.image_1_and_image_2_concat_feat_save_name))
    mm_feats = np.load(os.path.join(config.save_dir, config.multi_modal_feat_save_name))
    bert_feats1 = np.load(config.save_dir, config.bert_feature_save_name)

    image_knn_search(to_cuda, config, img_feats, mm_feats)
    nlp_knn_search(to_cuda, config, bert_feats1)


def main():
    config = Config(
        data_path="/2021-top-data-competition/1-shopee-goods-match-competition/shopee-product-matching/test.csv",
        image_dir_path="/2021-top-data-competition/1-shopee-goods-match-competition/shopee-product-matching/test_images",
        save_dir="temp")

    # config = Config(
    #     data_path="/2021-top-data-competition/1-shopee-goods-match-competition/shopee-product-matching/train.csv",
    #     image_dir_path="/2021-top-data-competition/1-shopee-goods-match-competition/shopee-product-matching/train_images",
    #     save_dir="temp")

    get_feature_from_pretrained_model(config=config, nrows=100, to_cuda=False)
    fasis_first_recall(to_cuda=False, config=config)


# main
if __name__ == '__main__':
    main()
