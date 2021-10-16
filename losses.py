import torch.nn.functional as F
import torch

def content_loss(src_feats, dst_feats):
    '''
    src_feats and dst_feats are features from the generated image and
    target content image
    The features are from some certain layers, which is appointed by yourself

    e.g. src_feats maybe the output features from the layer 5, 10, 13 of a vgg-16
    note that the target features should be drawn from the same appointed layers
    '''
    loss = 0
    nr_layer = len(src_feats)
    for n in range(nr_layer):
        loss += F.mse_loss(src_feats[n], dst_feats[n])
    return loss / nr_layer

def gram(x):
    '''
    gram matrix does not give a damn about spacial information
    In gram matrix, we take any two indices e.g. i and j across all channels
    we now have C x 1 and 1 x C tensors and take inner product of them. The result
    is scalar, and put it on index (i, j)
    for all indices we do the same operations (C x C times if you are curious)
    '''
    N, C, H, W = x.shape
    x = x.view(N * C, -1)
    return torch.mm(x, x.t()) / (H * W)

def style_loss(src_feats, dst_feats, weights):
    loss = 0
    nr_layer = len(src_feats)
    for n in range(nr_layer):
        gram_src = gram(src_feats[n])
        gram_dst = gram(dst_feats[n])
        loss += weights[n] * F.mse_loss(gram_src, gram_dst)
    return loss / nr_layer