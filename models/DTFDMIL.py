import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

"""
The batch size is set to be 1, 
    meaning that in each iteration,  one slide is processed

The code is copied from https://github.com/hrzhang1123/DTFD-MIL/tree/main
from paper Zhang, Hongrun et al. “DTFD-MIL: Double-Tier Feature Distillation Multiple Instance Learning for Histopathology Whole Slide Image Classification.” 2022 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) (2022): 18780-18790.
"""


class Classifier_1fc(nn.Module):
    def __init__(self, n_channels, n_classes, droprate=0.0):
        super(Classifier_1fc, self).__init__()
        self.fc = nn.Linear(n_channels, n_classes)
        self.droprate = droprate
        if self.droprate != 0.0:
            self.dropout = torch.nn.Dropout(p=self.droprate)

    def forward(self, x):

        if self.droprate != 0.0:
            x = self.dropout(x)
        x = self.fc(x)
        return x


class residual_block(nn.Module):
    def __init__(self, nChn=512):
        super(residual_block, self).__init__()
        self.block = nn.Sequential(
                nn.Linear(nChn, nChn, bias=False),
                nn.ReLU(inplace=True),
                nn.Linear(nChn, nChn, bias=False),
                nn.ReLU(inplace=True),
            )
    def forward(self, x):
        tt = self.block(x)
        x = x + tt
        return x


class DimReduction(nn.Module):
    def __init__(self, n_channels, m_dim=512, numLayer_Res=0):
        super(DimReduction, self).__init__()
        self.fc1 = nn.Linear(n_channels, m_dim, bias=False)
        self.relu1 = nn.ReLU(inplace=True)
        self.numRes = numLayer_Res

        self.resBlocks = []
        for ii in range(numLayer_Res):
            self.resBlocks.append(residual_block(m_dim))
        self.resBlocks = nn.Sequential(*self.resBlocks)

    def forward(self, x):

        x = self.fc1(x)
        x = self.relu1(x)

        if self.numRes > 0:
            x = self.resBlocks(x)

        return x


class Attention_Gated(nn.Module):
    def __init__(self, L=512, D=128, K=1):
        super(Attention_Gated, self).__init__()

        self.L = L
        self.D = D
        self.K = K

        self.attention_V = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.Tanh()
        )

        self.attention_U = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.Sigmoid()
        )

        self.attention_weights = nn.Linear(self.D, self.K)

    def forward(self, x, isNorm=True):
        ## x: N x L
        A_V = self.attention_V(x)  # NxD
        A_U = self.attention_U(x)  # NxD
        A = self.attention_weights(A_V * A_U) # NxK
        A = torch.transpose(A, 1, 0)  # KxN

        if isNorm:
            A = F.softmax(A, dim=1)  # softmax over N

        return A  ### K x N


class Attention_with_Classifier(nn.Module):
    def __init__(self, num_cls, L=512, D=128, K=1, droprate=0):
        super(Attention_with_Classifier, self).__init__()
        self.attention = Attention_Gated(L, D, K)
        self.classifier = Classifier_1fc(L, num_cls, droprate)
    
    def forward(self, x): ## x: N x L
        AA = self.attention(x)  ## K x N
        afeat = torch.mm(AA, x) ## K x L
        pred = self.classifier(afeat) ## K x num_cls
        return pred

class DTFDMIL(nn.Module):
    def __init__(self, in_chn, num_cls, **kwargs):
        super().__init__()

        # default values based on https://github.com/hrzhang1123/DTFD-MIL/blob/10964f4dcc27c65ce110a0e9a3b9240bff58da8a/Main_DTFD_MIL.py#L272
        self.numGroup = 3
        self.total_instance = 3

        # default values based on https://github.com/hrzhang1123/DTFD-MIL/blob/main/Main_DTFD_MIL.py    
        self.mDim = 512
        self.classifier = Classifier_1fc(self.mDim, n_classes=num_cls, droprate=0)
        self.attention = Attention_Gated(self.mDim)
        self.dimReduction = DimReduction(in_chn, self.mDim, numLayer_Res=0)
        self.attCls = Attention_with_Classifier(L=self.mDim, num_cls=num_cls, droprate=0)
    
    def forward(self, x, return_attn=False, **kwargs):
        """
        args:
            x (torch.Tensor): is a bag of feature vectors of shape (1,N,C) or (N,C) 
                        where N is the number of patches in the WSI.
            
            return_attn (bool): weither to return tiles attention scores
        """

        if len(x.shape) == 3:
            x = x.squeeze(dim=0)

        slide_pseudo_feat = []
        slide_sub_preds = []

        feat_index = list(range(x.shape[0]))
        random.shuffle(feat_index)
        index_chunk_list = np.array_split(np.array(feat_index), self.numGroup)
        index_chunk_list = [sst.tolist() for sst in index_chunk_list]

        for tindex in index_chunk_list:
            subFeat_tensor = torch.index_select(x, dim=0, index=torch.LongTensor(tindex).to(x.device))
            tmidFeat = self.dimReduction(subFeat_tensor)
            tAA = self.attention(tmidFeat).squeeze(0)
            tattFeats = torch.einsum('ns,n->ns', tmidFeat, tAA)  ### n x fs
            tattFeat_tensor = torch.sum(tattFeats, dim=0).unsqueeze(0)  ## 1 x fs
            tPredict = self.classifier(tattFeat_tensor)  ### 1 x 2
            slide_sub_preds.append(tPredict)
            slide_pseudo_feat.append(tattFeat_tensor)

        slide_pseudo_feat = torch.cat(slide_pseudo_feat, dim=0)  ### numGroup x fs
        slide_sub_preds = torch.cat(slide_sub_preds, dim=0) ### numGroup x fs
        gSlidePred = self.attCls(slide_pseudo_feat)

        if return_attn:
            return (slide_sub_preds, gSlidePred), None
        else:
            return slide_sub_preds, gSlidePred
    
    def calculate_loss(self, y_hat, y, loss_fn, **kwargs):
        slide_sub_preds, gSlidePred = y_hat
        sub_y = torch.full(slide_sub_preds.shape[:1], y.item(), dtype=y.dtype, device=y.device)
        loss0 = loss_fn(slide_sub_preds.squeeze(dim=-1), sub_y)
        
        loss1 = loss_fn(gSlidePred, y)
        
        return loss0 + loss1
