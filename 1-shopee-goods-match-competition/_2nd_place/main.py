# !/user/bin/python
# -*- coding:utf-8 -*-
"""
date：          2021-06-30
Description :
auther : wcy
"""
# import modules
from _2nd_place.feature.get_feature import get_image_and_multi_modal_features, get_nlp_features
from _2nd_place.config import Config
from _2nd_place.utils import load_author_pretrained_model_ckpt
import logging

__all__ = []


# define function
def get_feature_from_pretrained_model(config: Config):
    logging.info(" -- load image model and multi modal model to get feature -- ")
    image_model1_ckpt = load_author_pretrained_model_ckpt(config.image1_trained_moel_path)
    image_model2_ckpt = load_author_pretrained_model_ckpt(config.image2_trained_model_path)
    multi_modal_model_ckpt = load_author_pretrained_model_ckpt(config.multi_modal_trained_model_path)
    get_image_and_multi_modal_features(config=config, image_model1_ckpt=image_model1_ckpt,
                                       image_model2_ckpt=image_model2_ckpt,
                                       multi_modal_model_ckpt=multi_modal_model_ckpt, to_cuda=False, nrows=None)

    logging.info(" -- load bert model to get feature -- ")
    bert_model_ckpt = load_author_pretrained_model_ckpt(config.bert_trained_model_path)
    bert2_model_ckpt = load_author_pretrained_model_ckpt(config.bert2_trained_model_path)
    bert3_model_ckpt = load_author_pretrained_model_ckpt(config.bert3_trained_model_path)

    get_nlp_features(config=config, bert_model_ckpt=bert_model_ckpt, bert2_model_ckpt=bert2_model_ckpt,
                     bert3_model_ckpt=bert3_model_ckpt, nrows=None)


def main():
    config = Config(data_path="", image_dir_path="", save_dir="temp")
    get_feature_from_pretrained_model(config=config)


# main
if __name__ == '__main__':
    main()
