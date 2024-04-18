#quoccuong
import torch
import torch.nn.functional as F


def DA_loss(features,target):
    loss = 0

    for feature in features:
        N,C,H,W = feature.shape
        feature = feature.permute(0,2,3,1)
        label = torch.zeros_like(feature) if target == 0 else torch.ones_like(feature)
        feature_end = feature.reshape(N,-1)
        label_end = label.reshape(N,-1)
        _loss = F.binary_cross_entropy_with_logits(feature_end,label_end)
        loss += _loss
    return loss/len(features)