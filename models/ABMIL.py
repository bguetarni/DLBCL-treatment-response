import torch
import torch.nn as nn
import torch.nn.functional as F

"""
The batch size is set to be 1, 
    meaning that in each iteration,  one slide is processed

The code is copied from https://github.com/AMLab-Amsterdam/AttentionDeepMIL
from paper Ilse, M., Tomczak, J. M., & Welling, M. (2018). Attention-based Deep Multiple Instance Learning. arXiv preprint arXiv:1802.04712.
"""

class ABMIL(nn.Module):
    def __init__(self, in_chn, num_cls):
        super(ABMIL, self).__init__()

        # default values from https://github.com/AMLab-Amsterdam/AttentionDeepMIL/blob/master/model.py
        self.M = 500
        self.L = 128
        self.ATTENTION_BRANCHES = 1

        self.feature_extractor_part2 = nn.Sequential(
            nn.Linear(in_chn, self.M),
            nn.ReLU(),
        )

        self.attention_V = nn.Sequential(
            nn.Linear(self.M, self.L), # matrix V
            nn.Tanh()
        )

        self.attention_U = nn.Sequential(
            nn.Linear(self.M, self.L), # matrix U
            nn.Sigmoid()
        )

        self.attention_w = nn.Linear(self.L, self.ATTENTION_BRANCHES) # matrix w (or vector w if self.ATTENTION_BRANCHES==1)

        self.classifier = nn.Linear(self.M*self.ATTENTION_BRANCHES, num_cls)

    def forward(self, x, return_attn=False, **kwargs):
        """
        args:
            x (torch.Tensor): a bag of feature vectors of shape (1,N,C) or (N,C)
            return_attn (bool): weither to return tiles attention score
        """
        
        if len(x.shape) == 3:
            x = x.squeeze(dim=0)

        H = self.feature_extractor_part2(x)  # KxM

        A_V = self.attention_V(H)  # KxL
        A_U = self.attention_U(H)  # KxL
        A = self.attention_w(A_V * A_U) # element wise multiplication # KxATTENTION_BRANCHES
        A = torch.transpose(A, 1, 0)  # ATTENTION_BRANCHESxK
        A = F.softmax(A, dim=1)  # softmax over K

        Z = torch.mm(A, H)  # ATTENTION_BRANCHESxM

        logits = self.classifier(Z)

        if return_attn:
            return logits, A
        else:
            return logits

    def calculate_loss(self, y_hat, y, loss_fn, **kwargs):
        return loss_fn(y_hat, y)
