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
from _2nd_place.config import Config
from _2nd_place.feature import image_similarity_model
from _2nd_place.feature import multi_modal_similarity_model
from _2nd_place.feature import nlp_similarity_model
from _2nd_place.utils import load_data, ShopeeDataset, image_transformer, BertDataset
from transformers import BertConfig, BertModel, BertTokenizerFast
from transformers import AutoTokenizer, AutoModel, AutoConfig

__all__ = []


# define function
def get_image_and_multi_modal_features(csv_path, image_dir_path, image_model1_ckpt, image_model2_ckpt,
                                       multi_modal_model_ckpt, to_cuda=False, nrows=None):
    # load data
    df, img_dir = load_data(csv_path=csv_path, image_dir_path=image_dir_path, nrows=nrows)
    dataset = ShopeeDataset(df=df, img_dir=img_dir, transform=None)
    data_loader = DataLoader(dataset, batch_size=8, shuffle=False,
                             drop_last=False, pin_memory=True, num_workers=Config.NUM_WORKERS, collate_fn=lambda x: x)

    # get model
    model1 = image_similarity_model.create_model(
        model_name=Config.image_backbone_model_1, pretrained=False,
        fc_dim=Config.image_feature_model1_fc_dim, p=Config.image_feature_model1_p_eval,
        to_cuda=to_cuda, model_ckpt=image_model1_ckpt, if_train=False)

    model2 = image_similarity_model.create_model(
        model_name=Config.image_backbone_model_2, pretrained=False,
        fc_dim=Config.image_feature_model2_fc_dim, p=Config.image_feature_model2_p_eval,
        to_cuda=to_cuda, model_ckpt=image_model2_ckpt, if_train=False)

    model3 = multi_modal_similarity_model.create_model(
        model_name=Config.multi_modal_image_backbone_model,
        bert_vocab_file=Config.bert_vocab_file, bert_config_file=Config.bert_config_file,
        max_len=Config.text_max_len, fc_dim=Config.multi_modal_image_fc_dim, p=Config.multi_modal_image_p_eval,
        to_cuda=to_cuda, model_ckpt=multi_modal_model_ckpt, if_train=False)

    img_feats1 = []
    img_feats2 = []
    mm_feats = []
    img_hs = []
    img_ws = []
    st_sizes = []

    transform = image_transformer(image_size=Config.image_size)
    for batch in tqdm(data_loader, total=len(data_loader), miniters=None, ncols=55):
        img, title, h, w, st_size = list(zip(*batch))
        img = torch.cat([transform(x.to('cuda').float() / 255)[None] for x in img], axis=0)
        title = list(title)
        with torch.no_grad():
            feats_minibatch1 = model1.extract_feat(img)
            img_feats1.append(feats_minibatch1.cpu().numpy())
            feats_minibatch2 = model2.extract_feat(img)
            img_feats2.append(feats_minibatch2.cpu().numpy())
            feats_minibatch3 = model3.extract_feat(img, title)
            mm_feats.append(feats_minibatch3.cpu().numpy())
        img_hs.extend(list(h))
        img_ws.extend(list(w))
        st_sizes.extend(list(st_size))

    # image model 1 feature
    img_feats1 = np.concatenate(img_feats1)
    img_feats1 /= np.linalg.norm(img_feats1, 2, axis=1, keepdims=True)
    np.save('/tmp/img_feats1', img_feats1)

    # image model 2 feature
    img_feats2 = np.concatenate(img_feats2)
    img_feats2 /= np.linalg.norm(img_feats2, 2, axis=1, keepdims=True)
    np.save('/tmp/img_feats2', img_feats2)

    # multi modal modal feature
    mm_feats = np.concatenate(mm_feats)
    mm_feats /= np.linalg.norm(mm_feats, 2, axis=1, keepdims=True)
    np.save('/tmp/mm_feats', mm_feats)

    # concat image model 1 and image model 2 feature
    img_feats = np.concatenate([img_feats1 * 1.0, img_feats2 * 1.0], axis=1)
    img_feats /= np.linalg.norm(img_feats, 2, axis=1, keepdims=True)
    np.save('/tmp/img_feats', img_feats)

    # user Fasis to get knn result
    if to_cuda:
        res = faiss.StandardGpuResources()
        index_img = faiss.IndexFlatIP(Config.image_feature_model1_fc_dim + Config.image_feature_model2_fc_dim)
        index_img = faiss.index_cpu_to_gpu(res, 0, index_img)
        index_img.add(img_feats)
        similarities_img, indexes_img = index_img.search(img_feats, Config.k)
    else:
        pass

    joblib.dump([similarities_img, indexes_img], '/tmp/lyk_img_data.pkl')
    joblib.dump([st_sizes, img_hs, img_ws], '/tmp/lyk_img_meta_data.pkl')

    if to_cuda:
        res = faiss.StandardGpuResources()
        index_mm = faiss.IndexFlatIP(Config.multi_modal_image_fc_dim)
        index_mm = faiss.index_cpu_to_gpu(res, 0, index_mm)
        index_mm.add(mm_feats)
        similarities_mm, indexes_mm = index_mm.search(mm_feats, Config.k)
    else:
        pass

    joblib.dump([similarities_mm, indexes_mm], '/tmp/lyk_mm_data.pkl')


def get_nlp_features(csv_path, image_dir_path, to_cuda, nrows=None):
    # load data
    df, img_dir = load_data(csv_path=csv_path, image_dir_path=image_dir_path, nrows=nrows)
    data_loaders = DataLoader(BertDataset(df=df),
                              batch_size=Config.bert_batch_size, shuffle=False,
                              drop_last=False, pin_memory=True, num_workers=Config.NUM_WORKERS)

    # get model
    nlp_similarity_model.create_model_1(
        vocab_file_path=Config.bert_vocab_file, bert_config_file=Config.bert_config_file,
        max_len=Config.text_max_len, fc_dim=Config.bert_fc_dim, simple_mean=True,
        to_cuda=to_cuda, model_ckpt=Config.bert_model_ckpt, if_train=False)



    from transformers import AutoTokenizer, AutoModel, AutoConfig
    model_name = params_bert2['model_name']
    tokenizer = AutoTokenizer.from_pretrained('../input/bertmultilingual/')
    bert_config = AutoConfig.from_pretrained('../input/bertmultilingual/')
    bert_model = AutoModel.from_config(bert_config)
    model2 = BertNet(bert_model, num_classes=0, tokenizer=tokenizer, max_len=params_bert['max_len'], simple_mean=False,
                     fc_dim=params_bert['fc_dim'], s=params_bert['s'], margin=params_bert['margin'], loss=params_bert['loss'])
    model2 = model2.to('cuda')
    model2.load_state_dict(checkpoint2['model'], strict=False)
    model2.train(False)

    #########

    model_name = params_bert3['model_name']
    tokenizer = AutoTokenizer.from_pretrained('../input/bertxlm/')
    bert_config = AutoConfig.from_pretrained('../input/bertxlm/')
    bert_model = AutoModel.from_config(bert_config)
    model3 = BertNet(bert_model, num_classes=0, tokenizer=tokenizer, max_len=params_bert3['max_len'], simple_mean=False,
                     fc_dim=params_bert3['fc_dim'], s=params_bert3['s'], margin=params_bert3['margin'], loss=params_bert3['loss'])
    model3 = model3.to('cuda')
    model3.load_state_dict(checkpoint3['model'], strict=False)
    model3.train(False)

    bert_feats1 = []
    bert_feats2 = []
    bert_feats3 = []
    for i, title in tqdm(enumerate(data_loaders['valid']),
                         total=len(data_loaders['valid']), miniters=None, ncols=55):
        with torch.no_grad():
            bert_feats_minibatch = model.extract_feat(title)
            bert_feats1.append(bert_feats_minibatch.cpu().numpy())
            bert_feats_minibatch = model2.extract_feat(title)
            bert_feats2.append(bert_feats_minibatch.cpu().numpy())
            bert_feats_minibatch = model3.extract_feat(title)
            bert_feats3.append(bert_feats_minibatch.cpu().numpy())

    bert_feats1 = np.concatenate(bert_feats1)
    bert_feats1 /= np.linalg.norm(bert_feats1, 2, axis=1, keepdims=True)
    bert_feats2 = np.concatenate(bert_feats2)
    bert_feats2 /= np.linalg.norm(bert_feats2, 2, axis=1, keepdims=True)
    bert_feats3 = np.concatenate(bert_feats3)
    bert_feats3 /= np.linalg.norm(bert_feats3, 2, axis=1, keepdims=True)

    bert_feats = np.concatenate([bert_feats1, bert_feats2], axis=1)
    bert_feats /= np.linalg.norm(bert_feats, 2, axis=1, keepdims=True)

    res = faiss.StandardGpuResources()
    index_bert = faiss.IndexFlatIP(params_bert['fc_dim'])
    index_bert = faiss.index_cpu_to_gpu(res, 0, index_bert)
    index_bert.add(bert_feats1)
    similarities_bert, indexes_bert = index_bert.search(bert_feats1, k)

    np.save('/tmp/bert_feats1', bert_feats1)
    np.save('/tmp/bert_feats2', bert_feats2)
    np.save('/tmp/bert_feats3', bert_feats3)

    bert_feats = np.concatenate([bert_feats1, bert_feats2, bert_feats3], axis=1)
    bert_feats /= np.linalg.norm(bert_feats, 2, axis=1, keepdims=True)

    np.save('/tmp/bert_feats', bert_feats)

    joblib.dump([similarities_bert, indexes_bert], '/tmp/lyk_bert_data.pkl')



# main
if __name__ == '__main__':
    pass