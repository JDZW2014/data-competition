# !/user/bin/python
# -*- coding:utf-8 -*-
"""
date：          2021-06-30
Description :
    这个文件的作用就是提取了图像 图像+文本的特征
auther : wcy
"""
# import modules
import timm
from _2nd_place.utils import gem
import torch.nn as nn

__all__ = ["ShopeeNet", "create_model", "train_model"]


# define class
class ShopeeNet(nn.Module):

    def __init__(self,
                 backbone, fc_dim=512, p=3):
        super(ShopeeNet, self).__init__()

        self.backbone = backbone
        self.backbone.reset_classifier(num_classes=0)

        self.fc = nn.Linear(self.backbone.num_features, fc_dim)
        self.bn = nn.BatchNorm1d(fc_dim)
        self._init_params()
        self.p = p

    def _init_params(self):
        nn.init.xavier_normal_(self.fc.weight)
        nn.init.constant_(self.fc.bias, 0)
        nn.init.constant_(self.bn.weight, 1)
        nn.init.constant_(self.bn.bias, 0)

    def extract_feat(self, x):
        batch_size = x.shape[0]
        x = self.backbone.forward_features(x)
        if isinstance(x, tuple):
            x = (x[0] + x[1]) / 2
            x = self.bn(x)
        else:
            x = gem(x, p=self.p).view(batch_size, -1)
            x = self.fc(x)
            x = self.bn(x)
        return x

    def forward(self, x, label):
        feat = self.extract_feat(x)
        x = self.loss_module(feat, label)
        return x, feat


# define function
def create_model(model_name, pretrained, fc_dim, p, to_cuda=False, model_ckpt=None, if_train=False):
    backbone = timm.create_model(model_name=model_name, pretrained=pretrained)
    model = ShopeeNet(backbone, fc_dim=fc_dim)
    model.p = p

    if model_ckpt is not None:
        model.load_state_dict(model_ckpt, strict=False)
    if to_cuda:
        model = model.to('cuda')
    if if_train is False:
        model.train(False)
    return model


def train_model():
    pass


# main
if __name__ == '__main__':
    pass

