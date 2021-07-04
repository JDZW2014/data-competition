# !/user/bin/python
# -*- coding:utf-8 -*-
"""
date：          2021-06-30
Description :
auther : wcy
"""
# import modules
import os
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
import numpy as np
import faiss
import joblib
from _2nd_place.feature import image_similarity_model
from _2nd_place.feature import multi_modal_similarity_model
from _2nd_place.feature import nlp_similarity_model
from _2nd_place.utils import load_data, ShopeeDataset, image_transformer, BertDataset
from _2nd_place.config import Config
from _2nd_place.utils import save_model_extracted_feat
import logging

__all__ = ["get_image_and_multi_modal_features", "get_nlp_features", "image_knn_search", "nlp_knn_search"]


# define function
def get_image_and_multi_modal_features(config: Config, image_model1_ckpt, image_model2_ckpt, multi_modal_model_ckpt,
                                       to_cuda=False, nrows=None):
    # load data
    logging.info(" -- load data -- ")
    df, img_dir = load_data(csv_path=config.data_path, image_dir_path=config.image_dir_path, nrows=nrows)
    logging.info("df shape is {}".format(df.shape))
    dataset = ShopeeDataset(df=df, img_dir=img_dir, transform=None)
    data_loader = DataLoader(dataset, batch_size=config.image_batch_size, shuffle=False,
                             drop_last=False, pin_memory=True, num_workers=config.NUM_WORKERS, collate_fn=lambda x: x)

    # get model
    logging.info(" -- load model1 --")
    model1 = image_similarity_model.create_model(
        model_name=config.image_backbone_model_1, pretrained=False,
        fc_dim=config.image_feature_model1_fc_dim, p=config.image_feature_model1_p_eval,
        to_cuda=to_cuda, model_ckpt=image_model1_ckpt, if_train=False)
    del image_model1_ckpt

    logging.info(" -- load model2 -- ")
    model2 = image_similarity_model.create_model(
        model_name=config.image_backbone_model_2, pretrained=False,
        fc_dim=config.image_feature_model2_fc_dim, p=config.image_feature_model2_p_eval,
        to_cuda=to_cuda, model_ckpt=image_model2_ckpt, if_train=False)
    del image_model2_ckpt

    logging.info(" -- load model3 -- ")
    model3 = multi_modal_similarity_model.create_model(
        model_name=config.multi_modal_image_backbone_model,
        bert_vocab_file=config.bert_vocab_file, bert_config_file=config.bert_config_file,
        max_len=config.multi_modal_text_max_len, fc_dim=config.multi_modal_image_fc_dim, p=config.multi_modal_p_eval,
        to_cuda=to_cuda, model_ckpt=multi_modal_model_ckpt, if_train=False)
    del multi_modal_model_ckpt

    logging.info(" -- extract feature -- ")
    img_feats1 = []
    img_feats2 = []
    mm_feats = []
    img_hs = []
    img_ws = []
    st_sizes = []

    transform = image_transformer(image_size=config.test_size)
    for batch in tqdm(data_loader, total=len(data_loader), miniters=None, ncols=55):
        img, title, h, w, st_size = list(zip(*batch))
        if to_cuda:
            img = torch.cat([transform(x.to('cuda').float() / 255)[None] for x in img], axis=0)
        else:
            img = torch.cat([transform(x.float() / 255)[None] for x in img], axis=0)

        title = list(title)
        with torch.no_grad():
            # feature 1
            feats_minibatch1 = model1.extract_feat(img)
            if to_cuda:
                img_feats1.append(feats_minibatch1.cpu().numpy())
            else:
                img_feats1.append(feats_minibatch1.numpy())

            # feature 2
            feats_minibatch2 = model2.extract_feat(img)
            if to_cuda:
                img_feats2.append(feats_minibatch2.cpu().numpy())
            else:
                img_feats2.append(feats_minibatch2.numpy())
            # feature 3
            feats_minibatch3 = model3.extract_feat(img, title)
            if to_cuda:
                mm_feats.append(feats_minibatch3.cpu().numpy())
            else:
                mm_feats.append(feats_minibatch3.cpu().numpy())

        img_hs.extend(list(h))
        img_ws.extend(list(w))
        st_sizes.extend(list(st_size))

    logging.info(" -- normalize and concat feature -- ")
    # image model 1 feature
    img_feats1 = np.concatenate(img_feats1)
    img_feats1 /= np.linalg.norm(img_feats1, 2, axis=1, keepdims=True)
    save_model_extracted_feat(fs_path=os.path.join(config.save_dir, config.image_1_feat_save_name),
                              feat=img_feats1)
    # np.save(os.path.join(config.save_dir, config.image_1_feat_save_name), img_feats1)

    # image model 2 feature
    img_feats2 = np.concatenate(img_feats2)
    img_feats2 /= np.linalg.norm(img_feats2, 2, axis=1, keepdims=True)
    save_model_extracted_feat(fs_path=os.path.join(config.save_dir, config.image_2_feat_save_name),
                              feat=img_feats2)
    # fs = os.path.join(config.save_dir, config.image_2_feat_save_name)
    # assert not os.path.exists(fs)
    # np.save(fs, img_feats2)

    # multi modal modal feature
    mm_feats = np.concatenate(mm_feats)
    mm_feats /= np.linalg.norm(mm_feats, 2, axis=1, keepdims=True)
    save_model_extracted_feat(fs_path=os.path.join(config.save_dir, config.multi_modal_feat_save_name),
                              feat=mm_feats)
    # fs = os.path.join(config.save_dir, config.multi_modal_feat_save_name)
    # assert not os.path.exists(fs)
    # np.save(fs, mm_feats)

    # concat image model 1 and image model 2 feature
    img_feats = np.concatenate([img_feats1 * 1.0, img_feats2 * 1.0], axis=1)
    img_feats /= np.linalg.norm(img_feats, 2, axis=1, keepdims=True)
    save_model_extracted_feat(fs_path=os.path.join(config.save_dir, config.image_1_and_image_2_concat_feat_save_name),
                              feat=img_feats)
    # np.save(os.path.join(config.save_dir, 'img_feats'), img_feats)

    joblib.dump([st_sizes, img_hs, img_ws], os.path.join(config.save_dir, config.lyk_img_meta_data_save_name))


def image_knn_search(to_cuda, config, img_feats, mm_feats):
    index_img = faiss.IndexFlatIP(config.image_feature_model1_fc_dim + config.image_feature_model2_fc_dim)
    if to_cuda:
        res = faiss.StandardGpuResources()
        index_img = faiss.index_cpu_to_gpu(res, 0, index_img)
        index_img.add(img_feats)
        similarities_img, indexes_img = index_img.search(img_feats, config.k)  # 这个用的是KNN检索的相似商品
    else:
        index_img.add(img_feats)
        similarities_img, indexes_img = index_img.search(img_feats, config.k)

    joblib.dump([similarities_img, indexes_img], os.path.join(config.save_dir, config.lyk_img_data))

    index_mm = faiss.IndexFlatIP(config.multi_modal_image_fc_dim)
    if to_cuda:
        res = faiss.StandardGpuResources()
        index_mm = faiss.index_cpu_to_gpu(res, 0, index_mm)
        index_mm.add(mm_feats)
        similarities_mm, indexes_mm = index_mm.search(mm_feats, config.k)
    else:
        index_mm.add(mm_feats)
        similarities_mm, indexes_mm = index_mm.search(mm_feats, config.k)

    joblib.dump([similarities_mm, indexes_mm], os.path.join(config.save_dir, config.lyk_mm_data))


def get_nlp_features(config: Config, bert_model_ckpt, bert2_model_ckpt, bert3_model_ckpt, to_cuda=False, nrows=None):
    # load data
    logging.info(" -- load data -- ")
    df, img_dir = load_data(csv_path=config.data_path, image_dir_path=config.image_dir_path, nrows=nrows)
    logging.info("df shape is {}".format(df.shape))
    data_loaders = DataLoader(BertDataset(df=df),
                              batch_size=config.bert_batch_size, shuffle=False,
                              drop_last=False, pin_memory=False, num_workers=config.NUM_WORKERS)

    # get model
    logging.info(" -- load model1 -- ")
    model1 = nlp_similarity_model.create_model_1(
        vocab_file_path=config.bert_vocab_file, bert_config_file=config.bert_config_file,
        max_len=config.bert_max_len, fc_dim=config.bert_fc_dim, simple_mean=True,
        to_cuda=to_cuda, model_ckpt=bert_model_ckpt, if_train=False)
    del bert_model_ckpt

    logging.info(" -- load model2 -- ")
    model2 = nlp_similarity_model.create_model_2(pretrained_path=config.bert2_pretrained_path,
                                                 max_len=config.bert2_max_len, fc_dim=config.bert2_fc_dim,
                                                 simple_mean=False,
                                                 to_cuda=to_cuda, model_ckpt=bert2_model_ckpt, if_train=False)
    del bert2_model_ckpt

    logging.info(" -- load model3 -- ")
    model3 = nlp_similarity_model.create_model_2(pretrained_path=config.bert3_pretrained_path,
                                                 max_len=config.bert3_max_len, fc_dim=config.bert3_fc_dim,
                                                 simple_mean=False,
                                                 to_cuda=to_cuda, model_ckpt=bert3_model_ckpt, if_train=False)
    del bert3_model_ckpt

    logging.info(" -- extract feature --")
    bert_feats1 = []
    bert_feats2 = []
    bert_feats3 = []
    for i, title in tqdm(enumerate(data_loaders), total=len(data_loaders), miniters=None, ncols=55):
        with torch.no_grad():
            bert_feats_minibatch = model1.extract_feat(title)
            bert_feats1.append(bert_feats_minibatch.cpu().numpy())
            bert_feats_minibatch = model2.extract_feat(title)
            bert_feats2.append(bert_feats_minibatch.cpu().numpy())
            bert_feats_minibatch = model3.extract_feat(title)
            bert_feats3.append(bert_feats_minibatch.cpu().numpy())

    logging.info(" -- normalize and concat feature -- ")
    bert_feats1 = np.concatenate(bert_feats1)
    bert_feats1 /= np.linalg.norm(bert_feats1, 2, axis=1, keepdims=True)
    save_model_extracted_feat(fs_path=os.path.join(config.save_dir, config.bert_feature_save_name),
                              feat=bert_feats1)
    # np.save(os.path.join(config.save_dir, 'bert_feats1'), bert_feats1)

    bert_feats2 = np.concatenate(bert_feats2)
    bert_feats2 /= np.linalg.norm(bert_feats2, 2, axis=1, keepdims=True)
    save_model_extracted_feat(fs_path=os.path.join(config.save_dir, config.bert2_feature_save_name),
                              feat=bert_feats2)
    # np.save(os.path.join(config.save_dir, 'bert_feats2'), bert_feats2)

    bert_feats3 = np.concatenate(bert_feats3)
    bert_feats3 /= np.linalg.norm(bert_feats3, 2, axis=1, keepdims=True)
    save_model_extracted_feat(fs_path=os.path.join(config.save_dir, config.bert3_feature_save_name),
                              feat=bert_feats3)
    # np.save(os.path.join(config.save_dir, 'bert_feats3'), bert_feats3)

    bert_feats = np.concatenate([bert_feats1, bert_feats2, bert_feats3], axis=1)
    bert_feats /= np.linalg.norm(bert_feats, 2, axis=1, keepdims=True)
    save_model_extracted_feat(fs_path=os.path.join(config.save_dir, config.bert_model_feature_concat_save_name),
                              feat=bert_feats)
    # np.save(os.path.join(config.save_dir, 'bert_feats'), bert_feats)


def nlp_knn_search(to_cuda, config, bert_feats1):
    index_bert = faiss.IndexFlatIP(config.bert_fc_dim)
    if to_cuda:
        res = faiss.StandardGpuResources()
        index_bert = faiss.index_cpu_to_gpu(res, 0, index_bert)
        index_bert.add(bert_feats1)
        similarities_bert, indexes_bert = index_bert.search(bert_feats1, config.k)
    else:
        index_bert.add(bert_feats1)
        similarities_bert, indexes_bert = index_bert.search(bert_feats1, config.k)
    joblib.dump([similarities_bert, indexes_bert], os.path.join(config.save_dir, config.lyk_bert_data))


# main
if __name__ == '__main__':
    pass
